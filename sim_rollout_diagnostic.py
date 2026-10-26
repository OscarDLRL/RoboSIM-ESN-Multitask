#!/usr/bin/env python3
"""Physical ManiSkill rollout with grasp diagnostics.

Modes
-----
oracle:
    Execute original demonstration actions after state K-1.

esn:
    Execute ESN-predicted physical qpos as pd_joint_pos commands after K.

compare:
    Run oracle and ESN independently from the exact same saved state and
    produce comparable CSV/plots.

Diagnostics saved per step
--------------------------
- TCP XYZ
- cube XYZ
- TCP-to-cube XYZ distance
- TCP-to-cube XY distance
- vertical TCP/cube difference
- measured robot qpos
- mean finger qpos
- closing flag
- near-object flag
- ManiSkill success
- commanded ESN qpos when applicable

This script never uses future cube states to manipulate physics. Future
demonstration data are used only as oracle actions / reference diagnostics.
"""

import argparse
import csv
import importlib
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401

import rollout_config as rc


# ============================================================
# Generic helpers
# ============================================================

def to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def to_python_bool(value):
    arr = to_numpy(value)
    return bool(arr.reshape(-1)[0]) if arr.size == 1 else bool(arr.any())


def first_xyz(value):
    arr = to_numpy(value)
    if arr.ndim == 1:
        return np.asarray(arr[:3], np.float64)
    return np.asarray(arr.reshape(-1, arr.shape[-1])[0, :3], np.float64)


def get_robot_qpos(env):
    robot = env.unwrapped.agent.robot
    if hasattr(robot, "get_qpos"):
        q = robot.get_qpos()
    else:
        q = env.unwrapped.agent.get_qpos()
    q = to_numpy(q).reshape(-1, 9)[0]
    return np.asarray(q, np.float64)


def get_tcp_xyz(env):
    """Robustly retrieve Panda TCP world XYZ."""
    agent = env.unwrapped.agent

    # ManiSkill Panda agents normally expose agent.tcp.pose.p.
    tcp = getattr(agent, "tcp", None)
    if tcp is not None:
        pose = getattr(tcp, "pose", None)
        if pose is not None and hasattr(pose, "p"):
            return first_xyz(pose.p)

        if hasattr(tcp, "get_pose"):
            pose = tcp.get_pose()
            if hasattr(pose, "p"):
                return first_xyz(pose.p)

    # Useful fallback on some agent wrappers.
    tcp_pose = getattr(agent, "tcp_pose", None)
    if tcp_pose is not None and hasattr(tcp_pose, "p"):
        return first_xyz(tcp_pose.p)

    raise RuntimeError(
        "Could not retrieve TCP position. Inspect env.unwrapped.agent.tcp "
        "for this ManiSkill version."
    )


def get_cube_xyz(env):
    state = env.unwrapped.get_state_dict()
    cube = state["actors"]["cube"]
    return first_xyz(cube)


def rgb_from_obs(obs):
    try:
        image = obs["sensor_data"]["base_camera"]["rgb"]
    except (KeyError, TypeError):
        return None

    image = to_numpy(image)

    if image.ndim == 4:
        image = image[0]

    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    return image


# ============================================================
# Controller conversion
# ============================================================

def default_controller(env):
    controllers = env.unwrapped.agent.controller.controllers

    if "arm" not in controllers or "gripper" not in controllers:
        raise RuntimeError(
            "Expected composite Panda controller with arm and gripper"
        )

    return controllers["arm"], controllers["gripper"]


def to_action(qpos, env):
    """Convert physical Panda qpos to configured pd_joint_pos action."""
    q = np.asarray(qpos, np.float32).reshape(-1)

    if len(q) != 9 or not np.isfinite(q).all():
        raise ValueError("Expected 9 finite joint positions")

    arm, grip = default_controller(env)

    arm_target = q[:7].copy()
    arm_cfg = arm.config

    if bool(getattr(arm_cfg, "use_delta", False)):
        raise RuntimeError("Only absolute pd_joint_pos is supported")

    if bool(getattr(arm_cfg, "normalize_action", False)):
        lo = np.asarray(arm_cfg.lower, np.float32)
        hi = np.asarray(arm_cfg.upper, np.float32)

        arm_target = 2 * (arm_target - lo) / (hi - lo) - 1
        arm_target = np.clip(arm_target, -1, 1)

    finger = float(
        np.clip(
            np.mean(q[7:9]),
            -0.01,
            0.04,
        )
    )

    if bool(getattr(grip.config, "normalize_action", False)):
        low = np.asarray(grip.config.lower).reshape(-1)[0]
        high = np.asarray(grip.config.upper).reshape(-1)[0]

        grip_target = float(
            np.clip(
                2 * (finger - low) / (high - low) - 1,
                -1,
                1,
            )
        )
    else:
        grip_target = finger

    action = np.concatenate(
        [
            arm_target,
            [grip_target],
        ]
    ).astype(np.float32)

    if action.shape != env.action_space.shape[-1:]:
        raise RuntimeError(
            f"Computed action {action.shape}; env space {env.action_space.shape}"
        )

    return action


# ============================================================
# Load task/demo/model information
# ============================================================

def load_demo_context(task, trajectory, K):
    task_id = "PushCube-v1" if task == "push" else "PickCube-v1"

    demo = (
        Path.home()
        / ".maniskill"
        / "demos"
        / task_id
        / "motionplanning"
        / "trajectory.h5"
    )

    meta_path = demo.with_suffix(".json")

    with meta_path.open() as f:
        meta = json.load(f)

    episode = next(
        (
            e
            for e in meta["episodes"]
            if int(e["episode_id"]) == int(trajectory)
        ),
        None,
    )

    if episode is None:
        raise ValueError(
            f"No episode_id={trajectory} in {meta_path}"
        )

    control_mode = episode.get(
        "control_mode",
        meta["env_info"].get("env_kwargs", {}).get("control_mode"),
    )

    if control_mode != rc.CONTROL_MODE:
        raise RuntimeError(
            f"Original control mode is {control_mode!r}, "
            f"not {rc.CONTROL_MODE!r}"
        )

    with h5py.File(demo, "r") as f:
        group = f[f"traj_{trajectory}"]

        original_actions = np.asarray(
            group["actions"],
            np.float32,
        )

        if K < 1 or K >= len(original_actions) + 1:
            raise ValueError(
                f"K must be between 1 and {len(original_actions)}"
            )

        state_index = K - 1

        state_arrays = {
            kind: {
                key: np.asarray(dataset[state_index]).copy()
                for key, dataset in group["env_states"][kind].items()
            }
            for kind in group["env_states"].keys()
        }

        reference_final = np.asarray(
            group["env_states"]["actors"]["cube"][-1, :3],
            np.float64,
        )

        goal_actor = (
            "goal_region"
            if task == "push"
            else "goal_site"
        )

        goal_xyz = np.asarray(
            group["env_states"]["actors"][goal_actor][state_index, :3],
            np.float64,
        )

    return {
        "task_id": task_id,
        "demo": demo,
        "meta": meta,
        "episode": episode,
        "original_actions": original_actions,
        "state_index": state_index,
        "state_arrays": state_arrays,
        "reference_final": reference_final,
        "goal_xyz": goal_xyz,
    }


def load_esn_prediction(task, trajectory, K, neurons, run, split):
    import parameters as p
    from data_utils import load_split

    module = importlib.import_module(
        rc.ESN_MODULE
    )

    load_model = module.load_model
    predict_trajectory = module.predict_trajectory

    rows = load_split(
        split,
        data_path=p.DATA_PATH,
    )

    name = f"{task}_{trajectory:03d}"

    matches = [
        row
        for row in rows
        if row["id"] == name
    ]

    if not matches:
        raise ValueError(
            f"{name} not found in split={split}"
        )

    row = matches[0]

    model_path = (
        p.MODEL_DIR
        / f"N_{neurons}"
        / f"run_{run:02d}"
        / "model.npz"
    )

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    model = load_model(
        model_path
    )

    _, predicted_joints, score = predict_trajectory(
        model,
        row,
        K,
    )

    if predicted_joints is None or not len(predicted_joints):
        raise RuntimeError(
            "Current ESN does not predict joints"
        )

    print(
        f"ESN task score={score:.4f}; "
        f"predicted joints={predicted_joints.shape}; "
        f"model={model_path}"
    )

    return (
        np.asarray(predicted_joints, np.float32),
        float(score),
        model_path,
    )


# ============================================================
# Diagnostic row / rollout
# ============================================================

def diagnostic_row(
    env,
    mode,
    step,
    forecast_index,
    success,
    commanded_qpos=None,
):
    tcp = get_tcp_xyz(env)
    cube = get_cube_xyz(env)
    qpos = get_robot_qpos(env)

    delta = tcp - cube

    dist_xyz = float(
        np.linalg.norm(delta)
    )

    dist_xy = float(
        np.linalg.norm(delta[:2])
    )

    finger_mean = float(
        np.mean(qpos[7:9])
    )

    row = {
        "mode": mode,
        "step": int(step),
        "forecast_index": int(forecast_index),

        "tcp_x": float(tcp[0]),
        "tcp_y": float(tcp[1]),
        "tcp_z": float(tcp[2]),

        "cube_x": float(cube[0]),
        "cube_y": float(cube[1]),
        "cube_z": float(cube[2]),

        "tcp_minus_cube_x": float(delta[0]),
        "tcp_minus_cube_y": float(delta[1]),
        "tcp_minus_cube_z": float(delta[2]),

        "tcp_cube_dist_xyz": dist_xyz,
        "tcp_cube_dist_xy": dist_xy,
        "tcp_cube_abs_z": float(abs(delta[2])),

        "finger_left_qpos": float(qpos[7]),
        "finger_right_qpos": float(qpos[8]),
        "finger_mean_qpos": finger_mean,

        "closing": int(
            finger_mean <= rc.GRIPPER_CLOSING_THRESHOLD
        ),

        "near_object": int(
            dist_xyz <= rc.NEAR_OBJECT_DISTANCE_M
        ),

        "success": int(bool(success)),
    }

    for j in range(9):
        row[f"measured_q{j+1}"] = float(qpos[j])

    if commanded_qpos is None:
        for j in range(9):
            row[f"command_q{j+1}"] = np.nan
    else:
        command = np.asarray(
            commanded_qpos,
            np.float64,
        ).reshape(9)

        for j in range(9):
            row[f"command_q{j+1}"] = float(command[j])

    return row


def run_rollout(
    context,
    mode,
    predicted_joints=None,
    video=False,
    max_episode_steps=None,
    ignore_termination=False,
):
    task_id = context["task_id"]
    meta = context["meta"]
    episode = context["episode"]
    original_actions = context["original_actions"]
    state_index = context["state_index"]
    state_arrays = context["state_arrays"]

    env_kwargs = dict(
        meta["env_info"].get(
            "env_kwargs",
            {}
        )
    )

    env_kwargs.update(
        obs_mode=rc.OBS_MODE,
        control_mode=rc.CONTROL_MODE,
        sim_backend=rc.SIM_BACKEND,
        num_envs=1,
    )

    min_steps = (
        len(original_actions)
        - state_index
        + rc.TIME_LIMIT_MARGIN
    )

    time_limit = (
        max_episode_steps
        or max(
            rc.MIN_TIME_LIMIT,
            min_steps,
        )
    )

    if time_limit < min_steps:
        raise ValueError(
            f"--max-episode-steps must be >= {min_steps}"
        )

    env = gym.make(
        task_id,
        max_episode_steps=time_limit,
        **env_kwargs,
    )

    frames = []
    log_rows = []

    try:
        obs, info = env.reset(
            **episode["reset_kwargs"]
        )

        device = env.unwrapped.device

        state = {
            group: {
                name: torch.as_tensor(
                    value,
                    device=device,
                ).unsqueeze(0)
                for name, value in subtree.items()
            }
            for group, subtree in state_arrays.items()
        }

        env.unwrapped.set_state_dict(
            state
        )

        # Initial state diagnostic before any forecast action.
        initial_success = to_python_bool(
            info.get("success", False)
        )

        log_rows.append(
            diagnostic_row(
                env,
                mode=mode,
                step=-1,
                forecast_index=-1,
                success=initial_success,
                commanded_qpos=None,
            )
        )

        if video:
            try:
                image = env.render()
                image = to_numpy(image)

                if image.ndim == 4:
                    image = image[0]

                if image.shape[-1] == 3:
                    frames.append(
                        np.uint8(image)
                    )
            except Exception as exc:
                print(
                    "Initial rendering unavailable:",
                    exc,
                )

        n_future = (
            len(original_actions)
            - state_index
        )

        if mode == "esn":
            n_future = min(
                n_future,
                len(predicted_joints),
            )

        ever_success = False
        last_success = False
        executed = 0
        stop_reason = "forecast_exhausted"

        first_closing_step = None
        minimum_distance = np.inf
        minimum_distance_step = None

        for i in range(n_future):
            if mode == "oracle":
                action = original_actions[
                    state_index + i
                ]
                commanded_qpos = None
            else:
                commanded_qpos = predicted_joints[i]
                action = to_action(
                    commanded_qpos,
                    env,
                )

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            executed += 1

            success = to_python_bool(
                info.get("success", False)
            )

            last_success = success
            ever_success |= success

            diag = diagnostic_row(
                env,
                mode=mode,
                step=i,
                forecast_index=i,
                success=success,
                commanded_qpos=commanded_qpos,
            )

            log_rows.append(
                diag
            )

            if (
                first_closing_step is None
                and diag["closing"]
            ):
                first_closing_step = i

            if (
                diag["tcp_cube_dist_xyz"]
                < minimum_distance
            ):
                minimum_distance = (
                    diag["tcp_cube_dist_xyz"]
                )
                minimum_distance_step = i

            if video:
                image = rgb_from_obs(
                    obs
                )

                if image is not None:
                    frames.append(
                        image
                    )

            if to_python_bool(truncated):
                stop_reason = (
                    "truncated_by_environment"
                )
                break

            if (
                to_python_bool(terminated)
                and not ignore_termination
            ):
                stop_reason = (
                    "terminated_by_environment"
                )
                break

        final_xyz = get_cube_xyz(
            env
        )

        summary = {
            "mode": mode,
            "executed": int(executed),
            "n_future": int(n_future),
            "ever_success": bool(ever_success),
            "success_at_end": bool(last_success),
            "stop_reason": stop_reason,
            "first_closing_step":
                (
                    None
                    if first_closing_step is None
                    else int(first_closing_step)
                ),
            "minimum_tcp_cube_distance_m":
                float(minimum_distance),
            "minimum_distance_step":
                (
                    None
                    if minimum_distance_step is None
                    else int(minimum_distance_step)
                ),
            "final_cube_xyz":
                np.asarray(final_xyz).tolist(),
        }

        return summary, log_rows, frames

    finally:
        env.close()


# ============================================================
# Output
# ============================================================

def save_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_gif(path, frames):
    if not frames:
        return

    from PIL import Image

    images = [
        Image.fromarray(frame).resize(
            (
                rc.GIF_SIZE,
                rc.GIF_SIZE,
            )
        )
        for frame in frames
    ]

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=rc.GIF_DURATION_MS,
        loop=0,
    )


def save_plots(out_dir, stem, results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib unavailable; skipping plots"
        )
        return

    # --------------------------------------------------------
    # Plot 1: TCP-to-cube distance
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    for mode, (_, rows, _) in results.items():
        steps = [
            row["step"]
            for row in rows
            if row["step"] >= 0
        ]

        distance = [
            100.0 * row["tcp_cube_dist_xyz"]
            for row in rows
            if row["step"] >= 0
        ]

        plt.plot(
            steps,
            distance,
            label=mode,
        )

    plt.axhline(
        100.0 * rc.NEAR_OBJECT_DISTANCE_M,
        linestyle="--",
        label="near-object threshold",
    )

    plt.xlabel("Forecast step")
    plt.ylabel("TCP-cube distance [cm]")
    plt.title("TCP-to-cube distance")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{stem}_tcp_cube_distance.png",
        dpi=160,
    )
    plt.close()

    # --------------------------------------------------------
    # Plot 2: finger qpos
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    for mode, (_, rows, _) in results.items():
        steps = [
            row["step"]
            for row in rows
            if row["step"] >= 0
        ]

        finger = [
            row["finger_mean_qpos"]
            for row in rows
            if row["step"] >= 0
        ]

        plt.plot(
            steps,
            finger,
            label=mode,
        )

    plt.axhline(
        rc.GRIPPER_CLOSING_THRESHOLD,
        linestyle="--",
        label="closing threshold",
    )

    plt.xlabel("Forecast step")
    plt.ylabel("Mean finger qpos [m]")
    plt.title("Gripper closing timing")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{stem}_gripper.png",
        dpi=160,
    )
    plt.close()

    # --------------------------------------------------------
    # Plot 3: XY and Z TCP/cube error
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    for mode, (_, rows, _) in results.items():
        steps = [
            row["step"]
            for row in rows
            if row["step"] >= 0
        ]

        xy = [
            100.0 * row["tcp_cube_dist_xy"]
            for row in rows
            if row["step"] >= 0
        ]

        plt.plot(
            steps,
            xy,
            label=f"{mode} XY",
        )

    plt.xlabel("Forecast step")
    plt.ylabel("TCP-cube XY distance [cm]")
    plt.title("Horizontal grasp alignment")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{stem}_xy_alignment.png",
        dpi=160,
    )
    plt.close()


def print_summary(context, results):
    precision = int(rc.PRINT_PRECISION)

    print(
        f"\nTask={context['task_id']} "
        f"K={context['state_index'] + 1}"
    )

    print(
        "Original final cube xyz:",
        np.round(
            context["reference_final"],
            precision,
        ),
    )

    print(
        "Goal xyz:",
        np.round(
            context["goal_xyz"],
            precision,
        ),
    )

    for mode, (summary, _, _) in results.items():
        final_xyz = np.asarray(
            summary["final_cube_xyz"],
            np.float64,
        )

        goal = context["goal_xyz"]

        print(
            f"\n[{mode.upper()}]"
        )

        print(
            "Simulated final cube xyz:",
            np.round(
                final_xyz,
                precision,
            ),
        )

        print(
            "Final cube-to-goal XYZ [m]:",
            f"{np.linalg.norm(final_xyz - goal):.{precision}f}",
        )

        print(
            "Final cube-to-goal XY  [m]:",
            f"{np.linalg.norm(final_xyz[:2] - goal[:2]):.{precision}f}",
        )

        print(
            "Minimum TCP-cube distance [m]:",
            f"{summary['minimum_tcp_cube_distance_m']:.{precision}f}",
            "at step",
            summary["minimum_distance_step"],
        )

        print(
            "First closing step:",
            summary["first_closing_step"],
        )

        print(
            "Ever success=",
            summary["ever_success"],
            "| Success at end=",
            summary["success_at_end"],
            "| Executed=",
            f"{summary['executed']}/{summary['n_future']}",
        )

        print(
            "Stop reason:",
            summary["stop_reason"],
        )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--task",
        choices=["pick", "push"],
        default=rc.DEFAULT_TASK,
    )

    ap.add_argument(
        "--trajectory",
        type=int,
        default=rc.DEFAULT_TRAJECTORY,
    )

    ap.add_argument(
        "--K",
        type=int,
        default=rc.DEFAULT_K,
    )

    ap.add_argument(
        "--mode",
        choices=["oracle", "esn", "compare"],
        default=rc.DEFAULT_MODE,
    )

    ap.add_argument(
        "--neurons",
        type=int,
        default=rc.DEFAULT_NEURONS,
    )

    ap.add_argument(
        "--run",
        type=int,
        default=rc.DEFAULT_RUN,
    )

    ap.add_argument(
        "--split",
        choices=["validation", "train", "test"],
        default=rc.DEFAULT_SPLIT,
    )

    ap.add_argument(
        "--video",
        action="store_true",
        default=rc.SAVE_VIDEO_BY_DEFAULT,
    )

    ap.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--ignore-termination",
        action="store_true",
    )

    args = ap.parse_args()

    context = load_demo_context(
        args.task,
        args.trajectory,
        args.K,
    )

    predicted_joints = None

    if args.mode in (
        "esn",
        "compare",
    ):
        (
            predicted_joints,
            score,
            model_path,
        ) = load_esn_prediction(
            args.task,
            args.trajectory,
            args.K,
            args.neurons,
            args.run,
            args.split,
        )

        expected = (
            len(context["original_actions"])
            + 1
        )

        # Dataset alignment is checked indirectly by requested episode ID.
        print(
            f"Model score={score:.4f}; "
            f"source model={model_path}"
        )

    modes = (
        ["oracle", "esn"]
        if args.mode == "compare"
        else [args.mode]
    )

    results = {}

    for mode in modes:
        summary, rows, frames = run_rollout(
            context,
            mode=mode,
            predicted_joints=predicted_joints,
            video=args.video,
            max_episode_steps=args.max_episode_steps,
            ignore_termination=args.ignore_termination,
        )

        results[mode] = (
            summary,
            rows,
            frames,
        )

    print_summary(
        context,
        results,
    )

    out_dir = Path(
        rc.OUTPUT_DIR
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{args.task}_{args.trajectory:03d}_K{args.K}"
    )

    for mode, (_, rows, frames) in results.items():
        if rc.SAVE_CSV:
            csv_path = (
                out_dir
                / f"{stem}_{mode}_diagnostics.csv"
            )

            save_csv(
                csv_path,
                rows,
            )

            print(
                "Saved CSV:",
                csv_path,
            )

        if args.video:
            gif_path = (
                out_dir
                / f"{stem}_{mode}.gif"
            )

            save_gif(
                gif_path,
                frames,
            )

            print(
                "Saved GIF:",
                gif_path,
            )

    if rc.SAVE_PLOTS:
        save_plots(
            out_dir,
            stem,
            results,
        )

        print(
            "Saved diagnostic plots in:",
            out_dir,
        )


if __name__ == "__main__":
    main()
