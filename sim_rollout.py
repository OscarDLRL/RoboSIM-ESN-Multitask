"""Physical rollout for paired-intention dataset, organized by dataset/model/modality."""

import argparse, json, sys
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch
import mani_skill.envs

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from model_loader import load_model_package


ENV_ID = "IntentCubeRollout-v1"
EYE, TARGET = [0.60, -0.70, 0.60], [0.00, 0.00, 0.20]
PICK_TOL = PUSH_TOL = 0.015


@register_env(ENV_ID, max_episode_steps=220)
class IntentCubeRolloutEnv(PushCubeEnv):
    def __init__(self, *a, robot_uids="panda", robot_init_qpos_noise=0.0, **kw):
        super().__init__(
            *a,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kw,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=EYE, target=TARGET)
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=1.0,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=EYE, target=TARGET)
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=1.0,
            near=0.01,
            far=100,
        )

    def _load_scene(self, options):
        super()._load_scene(options)
        if self.goal_region not in self._hidden_objects:
            self._hidden_objects.append(self.goal_region)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)

        if "cube_xy" in options:
            xy = (
                torch.tensor(
                    options["cube_xy"],
                    dtype=torch.float32,
                    device=self.device,
                )
                .reshape(1, 2)
                .repeat(len(env_idx), 1)
            )

            xyz = torch.zeros(
                (len(env_idx), 3),
                dtype=torch.float32,
                device=self.device,
            )
            xyz[:, :2] = xy
            xyz[:, 2] = float(self.cube_half_size)

            self.obj.set_pose(
                Pose.create_from_pq(
                    p=xyz,
                    q=[1, 0, 0, 0],
                )
            )

        far = (
            torch.tensor(
                [10.0, 10.0, -1.0],
                dtype=torch.float32,
                device=self.device,
            )
            .reshape(1, 3)
            .repeat(len(env_idx), 1)
        )

        self.goal_region.set_pose(
            Pose.create_from_pq(
                p=far,
                q=[1, 0, 0, 0],
            )
        )

    def evaluate(self):
        return {
            "success": torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        }


def npy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    a = np.asarray(x)
    return a[0] if a.ndim and a.shape[0] == 1 else a


def qpos(env):
    return npy(env.unwrapped.agent.robot.get_qpos()).astype(np.float32)


def cube(env):
    return npy(env.unwrapped.obj.pose.p).astype(np.float32)


def rgb(obs):
    try:
        a = npy(obs["sensor_data"]["base_camera"]["rgb"])

        if a.dtype != np.uint8:
            if np.issubdtype(a.dtype, np.floating) and a.max() <= 1.01:
                a = a * 255

            a = np.clip(a, 0, 255).astype(np.uint8)

        return a

    except (KeyError, TypeError):
        return None


def qpos_to_action(q, env):
    q = np.asarray(q, np.float32).reshape(-1)

    if q.shape != (9,) or not np.isfinite(q).all():
        raise ValueError("Expected 9 finite qpos")

    c = env.unwrapped.agent.controller.controllers
    arm, grip = c["arm"], c["gripper"]

    if arm.config.use_delta:
        raise RuntimeError("Expected absolute pd_joint_pos")

    a = q[:7].copy()

    if arm.config.normalize_action:
        lo = np.asarray(arm.config.lower, np.float32)
        hi = np.asarray(arm.config.upper, np.float32)
        a = np.clip(2 * (a - lo) / (hi - lo) - 1, -1, 1)

    f = float(np.clip(np.mean(q[7:9]), -0.01, 0.04))

    if grip.config.normalize_action:
        lo = float(np.asarray(grip.config.lower).reshape(-1)[0])
        hi = float(np.asarray(grip.config.upper).reshape(-1)[0])
        f = float(np.clip(2 * (f - lo) / (hi - lo) - 1, -1, 1))

    action = np.r_[a, f].astype(np.float32)

    if action.shape != env.action_space.shape[-1:]:
        raise RuntimeError(
            f"Action {action.shape}; expected {env.action_space.shape}"
        )

    return action


def task_metrics(task, initial, final, target):
    initial, final, target = map(
        lambda x: np.asarray(x, float),
        (initial, final, target),
    )

    if task == "pick":
        err = float(np.linalg.norm(final - target))
        lift = float(final[2] - initial[2])
        desired = float(target[2] - initial[2])

        return dict(
            success=bool(
                err <= PICK_TOL
                and lift >= desired - PICK_TOL
            ),
            target_error=err,
            primary_displacement=lift,
            lateral_error=float(
                np.linalg.norm(final[:2] - initial[:2])
            ),
            z_error=float(abs(final[2] - target[2])),
        )

    err = float(np.linalg.norm(final[:2] - target[:2]))
    dx, dy, dz = final - initial
    desired = float(target[0] - initial[0])

    return dict(
        success=bool(
            err <= PUSH_TOL
            and abs(dz) <= 0.012
            and dx >= desired - PUSH_TOL
        ),
        target_error=err,
        primary_displacement=float(dx),
        lateral_error=float(abs(dy)),
        z_error=float(abs(dz)),
    )


def modality_tag(p, mode):
    """
    Folder/metadata tag for the actual input condition used at rollout.

    Oracle does not use the ESN input mask, so it gets its own tag.
    """
    if mode == "oracle":
        return "oracle"

    test_mode = getattr(p, "MODALITY_TEST_MODE", "full")

    mapping = {
        "full": "full",
        "vision": "vision_only",
        "joints": "joints_only",
    }

    if test_mode not in mapping:
        raise ValueError(
            "MODALITY_TEST_MODE must be 'full', 'vision' or 'joints'"
        )

    return mapping[test_mode]


def main():
    ap = argparse.ArgumentParser()

    # Clean project: one model package.
    ap.add_argument("--model", choices=("baseline","motion_only"), default="baseline")

    ap.add_argument("--task", choices=("pick", "push"), default="pick")
    ap.add_argument("--trajectory", type=int, default=6)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--mode", choices=("oracle", "esn"), default="esn")
    ap.add_argument("--neurons", type=int, default=4000)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--handoff-q-tol", type=float, default=2e-3)
    ap.add_argument("--handoff-cube-tol", type=float, default=2e-3)

    a = ap.parse_args()

    p, esn = load_model_package(a.model)
    sys.modules["parameters"] = p
    sys.modules["esn"] = esn

    dataset = Path(p.DATA_PATH).resolve()
    name = f"{a.task}_{a.trajectory:03d}"
    input_condition = modality_tag(p, a.mode)

    with h5py.File(dataset, "r") as h5:
        g = h5[f"trajectories/{name}"]

        split = g.attrs["split"]
        split = split.decode() if isinstance(split, bytes) else str(split)

        if split != a.split:
            raise ValueError(
                f"{name} belongs to {split}, not {a.split}"
            )

        actions = np.asarray(g["actions"], np.float32)
        qref = np.asarray(g["qpos"], np.float32)
        cref = np.asarray(g["cube_xyz"], np.float32)

        initial = np.asarray(
            g["initial_cube_xyz"],
            np.float32,
        )
        target = np.asarray(
            g["evaluation_target_xyz"],
            np.float32,
        )

    if not 1 <= a.K < len(qref):
        raise ValueError(
            f"K must be 1..{len(qref) - 1}"
        )

    pred_q = None
    score = None

    if a.mode == "esn":
        from data_utils import load_split

        row = next(
            r
            for r in load_split(
                a.split,
                data_path=p.DATA_PATH,
            )
            if r["id"] == name
        )

        model_path = (
            p.MODEL_DIR
            / f"N_{a.neurons}"
            / f"run_{a.run:02d}"
            / "model.npz"
        )

        model = esn.load_model(model_path)

        # Explicitly pass the test modality instead of relying only on
        # the global default inside predict_trajectory().
        test_mode = getattr(
            p,
            "MODALITY_TEST_MODE",
            "full",
        )

        _, pred_q, score = esn.predict_trajectory(
            model,
            row,
            a.K,
            modality_mode=test_mode,
        )

        if pred_q is None or not len(pred_q):
            raise RuntimeError(
                "Model does not predict joints"
            )

        print(
            f"MODEL={a.model} | "
            f"INPUT={input_condition} | "
            f"score={score:.4f} | "
            f"predicted={pred_q.shape}"
        )

    env = gym.make(
        ENV_ID,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        num_envs=1,
        max_episode_steps=max(
            220,
            len(actions) + 20,
        ),
    )

    frames = []

    try:
        obs, _ = env.reset(
            seed=42,
            options={
                "cube_xy": initial[:2].tolist()
            },
        )

        if a.video:
            im = rgb(obs)
            if im is not None:
                frames.append(im)

        # Replay the true prefix exactly.
        for i in range(a.K - 1):
            obs, _, _, trunc, _ = env.step(
                actions[i]
            )

            if a.video:
                im = rgb(obs)
                if im is not None:
                    frames.append(im)

            if bool(npy(trunc)):
                raise RuntimeError(
                    "Truncated during handoff"
                )

        qerr = float(
            np.max(
                np.abs(
                    qpos(env) - qref[a.K - 1]
                )
            )
        )

        cerr = float(
            np.max(
                np.abs(
                    cube(env) - cref[a.K - 1]
                )
            )
        )

        print(
            f"HANDOFF {name} K={a.K}: "
            f"q={qerr:.8f}, "
            f"cube={cerr:.8f}"
        )

        if qerr > a.handoff_q_tol:
            raise RuntimeError(
                f"q handoff error {qerr}"
            )

        if cerr > a.handoff_cube_tol:
            raise RuntimeError(
                f"cube handoff error {cerr}"
            )

        first = a.K - 1
        available = len(actions) - first

        if a.mode == "oracle":
            nfuture = available
        else:
            nfuture = min(
                available,
                len(pred_q),
            )

        executed = 0
        stop = "forecast_exhausted"

        for i in range(nfuture):
            if a.mode == "oracle":
                action = actions[first + i]
            else:
                action = qpos_to_action(
                    pred_q[i],
                    env,
                )

            obs, _, _, trunc, _ = env.step(action)
            executed += 1

            if a.video:
                im = rgb(obs)
                if im is not None:
                    frames.append(im)

            if bool(npy(trunc)):
                stop = "truncated"
                break

        final = cube(env)

    finally:
        env.close()

    summary = dict(
        dataset=dataset.parent.name,
        model=a.model,

        # Architecture mode remains dropout.
        experiment_mode=p.EXPERIMENT_MODE,

        # Actual modality condition used during this rollout.
        input_condition=input_condition,
        modality_test_mode=(
            None
            if a.mode == "oracle"
            else getattr(
                p,
                "MODALITY_TEST_MODE",
                "full",
            )
        ),

        architecture=esn.ARCHITECTURE,
        task=a.task,
        scene=a.trajectory,
        split=a.split,
        K=a.K,
        mode=a.mode,
        neurons=a.neurons,
        run=a.run,

        predicted_class=(
            None
            if score is None
            else (
                "push"
                if score >= 0.5
                else "pick"
            )
        ),

        class_score=(
            None
            if score is None
            else float(score)
        ),

        initial_cube_xyz=initial.tolist(),
        target_xyz=target.tolist(),
        recorded_final_xyz=cref[-1].tolist(),
        simulated_final_xyz=final.tolist(),

        **task_metrics(
            a.task,
            initial,
            final,
            target,
        ),

        executed_future=executed,
        available_future=nfuture,
        stop_reason=stop,
        handoff_q_error=qerr,
        handoff_cube_error_m=cerr,
    )

    print("\n" + "=" * 62)

    for k, v in summary.items():
        print(f"{k:24s}: {v}")

    print("=" * 62)

    # IMPORTANT:
    # full / vision_only / joints_only are now separate directories,
    # preventing one modality from overwriting another.
    out = (
        Path("sim_rollouts_intent")
        / dataset.parent.name
        / a.model
        / p.EXPERIMENT_MODE
        / input_condition
        / a.split
        / a.task
        / f"scene_{a.trajectory:03d}"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"K{a.K:02d}_"
        f"{a.mode}_"
        f"N{a.neurons}_"
        f"run{a.run:02d}"
    )

    js = out / f"{stem}.json"

    js.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "Saved summary:",
        js.resolve(),
    )

    if a.video and frames:
        from PIL import Image

        gif = out / f"{stem}.gif"

        imgs = [
            Image.fromarray(x).resize(
                (384, 384)
            )
            for x in frames
        ]

        imgs[0].save(
            gif,
            save_all=True,
            append_images=imgs[1:],
            duration=120,
            loop=0,
        )

        print(
            "Saved GIF:",
            gif.resolve(),
        )


if __name__ == "__main__":
    main()