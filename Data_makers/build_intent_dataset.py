#!/usr/bin/env python3
"""
Build paired-intention ManiSkill dataset (v5 large asymmetric workspace + spatially stratified split).

Design goals
------------
Each accepted scene produces TWO demonstrations from the SAME initial scene:

    pick:
        approach from above -> grasp -> lift cube by a fixed +Z distance

    push:
        approach low/horizontally from behind -> contact -> push cube by a
        fixed +X distance

Important anti-shortcut constraints
-----------------------------------
- Same environment, cube position, robot initial state, camera and appearance
  for the paired Pick/Push scene.
- No visible target / bullseye.
- The evaluation goal exists only as metadata. It is NOT rendered.
- interaction_mask has identical semantics in both classes: the cube.
- Split is assigned by SCENE, so the Pick/Push pair can never be separated
  across train/validation/test.
- The first RGB/qpos/cube observation of each pair is checked for equality.
  Therefore K=1 classification should be near chance if there are no hidden
  shortcuts; class information should emerge from the MOTOR INTENTION.

Camera
------
Sensor camera is the requested "diagonal_right" view:
    eye    = [0.60, -0.70, 0.60]
    target = [0.00,  0.00, 0.20]

Output schema is compatible with the existing data_utils.py:
    rgb
    qpos
    goal_xyz                # evaluation-only fixed target; keep USE_GOAL=False
    cube_xyz
    agent_mask
    interaction_mask
    offline_normalized_indices

Additional fields:
    tcp_xyz
    actions
    initial_cube_xyz
    evaluation_target_xyz

Requires ManiSkill + mplib (installed with standard ManiSkill motion-planning setup).

Recommended first run:
    python3 build_intent_dataset.py --count 5 \
        --out-dir datasets/maniskill_intent_smoke

Then inspect videos/metrics and, once satisfied:
    python3 build_intent_dataset.py --count 100 --grid-side 10 \
        --out-dir datasets/maniskill_intent_v1_final_new
"""

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


ENV_ID = "IntentCube-v1"

CAMERA_EYE = [0.60, -0.70, 0.60]
CAMERA_TARGET = [0.00, 0.00, 0.20]

PICK_LABEL = 0
PUSH_LABEL = 1

AGENT_NAME_TOKENS = ("panda",)
INTERACTION_NAMES = ("cube",)
GOAL_NAME_TOKENS = ("goal",)


# =====================================================================
# Custom task: same physical scene for both intentions, no visible goal
# =====================================================================

@register_env(ENV_ID, max_episode_steps=160)
class IntentCubeEnv(PushCubeEnv):
    """Shared scene used to generate both Pick and Push intentions."""

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.0,
        **kwargs,
    ):
        # Remove random robot posture as a possible class cue.
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=CAMERA_EYE,
            target=CAMERA_TARGET,
        )
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
        pose = sapien_utils.look_at(
            eye=CAMERA_EYE,
            target=CAMERA_TARGET,
        )
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=1.0,
            near=0.01,
            far=100,
        )

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        # PushCube normally has a red/white bullseye. It MUST NOT be a class cue.
        # ManiSkill supports hiding actors from visual sensor observations.
        if self.goal_region not in self._hidden_objects:
            self._hidden_objects.append(self.goal_region)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        b = len(env_idx)

        # Explicit paired cube position supplied by the dataset generator.
        cube_xy = options.get("cube_xy", None)
        if cube_xy is not None:
            xy = torch.as_tensor(
                cube_xy,
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 2).repeat(b, 1)

            xyz = torch.zeros(
                (b, 3),
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

        # Put the task's original visual goal physically far away too.
        # We do NOT use it for our success criterion.
        far = torch.tensor(
            [10.0, 10.0, -1.0],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 3).repeat(b, 1)

        self.goal_region.set_pose(
            Pose.create_from_pq(
                p=far,
                q=[1, 0, 0, 0],
            )
        )

    def evaluate(self):
        # Dataset success is computed ourselves from fixed physical rules.
        # Disable task termination from the original PushCube goal.
        return {
            "success": torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        }


# =====================================================================
# Tensor / observation helpers
# =====================================================================

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def unbatch(x):
    x = to_numpy(x)
    if x.ndim > 0 and x.shape[0] == 1:
        x = x[0]
    return x


def observation_rgb(obs):
    x = unbatch(
        obs["sensor_data"]["base_camera"]["rgb"]
    )

    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating):
            if float(np.nanmax(x)) <= 1.0 + 1e-6:
                x = np.clip(x, 0, 1) * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)

    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Unexpected RGB shape {x.shape}")

    return x


def observation_segmentation(obs):
    x = unbatch(
        obs["sensor_data"]["base_camera"]["segmentation"]
    )

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    if x.ndim != 2:
        raise ValueError(
            f"Unexpected segmentation shape {x.shape}"
        )

    return x.astype(np.int32)


def get_qpos(env):
    q = unbatch(
        env.unwrapped.agent.robot.get_qpos()
    ).astype(np.float32)

    if q.shape != (9,):
        raise ValueError(f"Expected Panda qpos (9,), got {q.shape}")

    return q


def get_cube_xyz(env):
    return unbatch(
        env.unwrapped.obj.pose.p
    ).astype(np.float32)


def get_tcp_xyz(env):
    return unbatch(
        env.unwrapped.agent.tcp.pose.p
    ).astype(np.float32)


# =====================================================================
# Capture wrapper: records exactly what the motion planner executes
# =====================================================================

class CaptureEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.clear()

    def clear(self):
        self.rgb = []
        self.segmentation = []
        self.qpos = []
        self.cube_xyz = []
        self.tcp_xyz = []
        self.actions = []

    def _append_observation(self, obs):
        self.rgb.append(observation_rgb(obs))
        self.segmentation.append(observation_segmentation(obs))
        self.qpos.append(get_qpos(self))
        self.cube_xyz.append(get_cube_xyz(self))
        self.tcp_xyz.append(get_tcp_xyz(self))

    def reset(self, **kwargs):
        self.clear()
        obs, info = self.env.reset(**kwargs)
        self._append_observation(obs)
        return obs, info

    def step(self, action):
        action_np = unbatch(action).astype(np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.actions.append(action_np)
        self._append_observation(obs)
        return obs, reward, terminated, truncated, info

    def episode(self):
        if not self.rgb:
            raise RuntimeError("No captured episode")

        if len(self.actions) + 1 != len(self.rgb):
            raise RuntimeError(
                f"actions={len(self.actions)} observations={len(self.rgb)}"
            )

        return {
            "rgb": np.stack(self.rgb).astype(np.uint8),
            "segmentation": np.stack(self.segmentation).astype(np.int32),
            "qpos": np.stack(self.qpos).astype(np.float32),
            "cube_xyz": np.stack(self.cube_xyz).astype(np.float32),
            "tcp_xyz": np.stack(self.tcp_xyz).astype(np.float32),
            "actions": np.stack(self.actions).astype(np.float32),
        }


# =====================================================================
# Semantic segmentation -> generic masks
# =====================================================================

def resolve_segmentation_ids(env):
    mapping = env.unwrapped.segmentation_id_map

    agent_ids = []
    interaction_ids = []
    goal_ids = []

    print("\nSEGMENTATION MAP")
    for obj_id, obj in sorted(mapping.items(), key=lambda x: int(x[0])):
        obj_id = int(obj_id)
        name = str(getattr(obj, "name", ""))
        lname = name.lower()

        print(
            f"  ID={obj_id:3d} "
            f"type={type(obj).__name__:15s} "
            f"name={name}"
        )

        if any(t in lname for t in AGENT_NAME_TOKENS):
            agent_ids.append(obj_id)

        if lname in INTERACTION_NAMES:
            interaction_ids.append(obj_id)

        if any(t in lname for t in GOAL_NAME_TOKENS):
            goal_ids.append(obj_id)

    if not agent_ids:
        raise RuntimeError("Could not resolve Panda segmentation IDs")
    if not interaction_ids:
        raise RuntimeError("Could not resolve cube segmentation ID")

    print("agent_ids      =", agent_ids)
    print("interaction_ids=", interaction_ids)
    print("goal_ids       =", goal_ids)

    return (
        np.asarray(agent_ids, np.int32),
        np.asarray(interaction_ids, np.int32),
        np.asarray(goal_ids, np.int32),
    )


def make_masks(segmentation, agent_ids, interaction_ids):
    agent = np.isin(
        segmentation,
        agent_ids,
    ).astype(np.uint8)

    interaction = np.isin(
        segmentation,
        interaction_ids,
    ).astype(np.uint8)

    if int(agent.sum()) == 0:
        raise ValueError("agent_mask is empty")

    if int(interaction.sum()) == 0:
        raise ValueError("interaction_mask is empty")

    return agent, interaction


def assert_goal_not_visible(segmentation, goal_ids):
    if len(goal_ids) == 0:
        return

    visible = np.isin(
        segmentation,
        goal_ids,
    )

    if bool(visible.any()):
        raise RuntimeError(
            "Goal/bullseye is visible in segmentation. "
            "This would be a task shortcut."
        )


# =====================================================================
# Motion-planning behaviors
# =====================================================================

def new_planner(env, vis=False):
    return PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )


def planner_step_ok(result):
    if result is None:
        return False
    if isinstance(result, (int, np.integer)):
        return int(result) != -1
    return True


def run_pick(
    env,
    scene_seed,
    cube_xy,
    lift_distance,
    pregrasp_height,
    vis=False,
):
    """
    Pick policy with clean geometric intent:

        identical initial state
              |
              | 1) go UP only
              v
        safe-height pose
              |
              | 2) move in XY / orient while staying high
              v
        vertically aligned above cube
              |
              | 3) ONE continuous vertical descent
              v
        grasp
              |
              | 4) fixed vertical lift
              v

    The important design constraint is that the TCP is not allowed to start
    descending toward the cube until its XY position is already aligned with
    the grasp point.
    """
    env.reset(
        seed=int(scene_seed),
        options={
            "cube_xy": np.asarray(cube_xy, np.float32).tolist(),
        },
    )

    planner = new_planner(env, vis=vis)
    e = env.unwrapped

    try:
        # -------------------------------------------------------------
        # Compute the final grasp pose FIRST.
        # -------------------------------------------------------------
        obb = get_actor_obb(e.obj)

        approaching = np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float64,
        )

        target_closing = (
            e.agent.tcp.pose
            .to_transformation_matrix()[0, :3, 1]
            .cpu()
            .numpy()
        )

        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )

        grasp_pose = e.agent.build_grasp_pose(
            approaching,
            grasp_info["closing"],
            e.obj.pose.sp.p,
        )

        grasp_p = np.asarray(
            grasp_pose.p,
            dtype=np.float64,
        )

        tcp0_p = np.asarray(
            e.agent.tcp.pose.sp.p,
            dtype=np.float64,
        )
        tcp0_q = np.asarray(
            e.agent.tcp.pose.sp.q,
            dtype=np.float64,
        )

        # -------------------------------------------------------------
        # 1) PURE UPWARD MOTION.
        #
        # Choose a safe height above BOTH the current TCP and the eventual
        # grasp point. This guarantees that Pick never starts descending
        # before it is horizontally aligned.
        # -------------------------------------------------------------
        safe_z = max(
            float(tcp0_p[2]) + 0.03,
            float(grasp_p[2]) + float(pregrasp_height),
        )

        up_pose = sapien.Pose(
            p=np.array(
                [
                    float(tcp0_p[0]),
                    float(tcp0_p[1]),
                    safe_z,
                ],
                dtype=np.float64,
            ),
            q=tcp0_q,
        )

        if not planner_step_ok(
            planner.move_to_pose_with_screw(up_pose)
        ):
            print("  PICK failed at stage: vertical-rise")
            return None

        # -------------------------------------------------------------
        # 2) HORIZONTAL ALIGNMENT ABOVE THE CUBE.
        #
        # X/Y are now exactly the grasp X/Y, while Z remains at safe_z.
        # Orientation is also changed here, while the robot is still high.
        # -------------------------------------------------------------
        aligned_pose = sapien.Pose(
            p=np.array(
                [
                    float(grasp_p[0]),
                    float(grasp_p[1]),
                    safe_z,
                ],
                dtype=np.float64,
            ),
            q=grasp_pose.q,
        )

        if not planner_step_ok(
            planner.move_to_pose_with_screw(aligned_pose)
        ):
            print("  PICK failed at stage: horizontal-align-above-cube")
            return None

        # -------------------------------------------------------------
        # 3) SINGLE CONTINUOUS DESCENT.
        #
        # No intermediate pregrasp waypoint.  This removes the visible
        # "descend -> pause -> descend again" artifact.
        # -------------------------------------------------------------
        if not planner_step_ok(
            planner.move_to_pose_with_screw(grasp_pose)
        ):
            print("  PICK failed at stage: continuous-vertical-descent")
            return None

        # The pause here is intentional and semantically meaningful:
        # the robot has reached the cube and closes the gripper.
        planner.close_gripper()

        # -------------------------------------------------------------
        # 4) FIXED VERTICAL LIFT.
        # -------------------------------------------------------------
        lift_pose = sapien.Pose(
            p=grasp_p + np.array(
                [0.0, 0.0, float(lift_distance)],
                dtype=np.float64,
            ),
            q=grasp_pose.q,
        )

        if not planner_step_ok(
            planner.move_to_pose_with_screw(lift_pose)
        ):
            print("  PICK failed at stage: fixed-lift")
            return None

        return env.episode()

    finally:
        planner.close()

def run_push(
    env,
    scene_seed,
    cube_xy,
    push_distance,
    push_tcp_travel,
    prepush_distance,
    contact_offset,
    push_tcp_z,
    vis=False,
):
    """
    Smooth low/table-parallel Push policy.

    There is no separate artificial "early cue" waypoint anymore.

    The first actual movement IS the intent:
        initial -> low pose behind cube -> contact -> push

    The gripper is not closed at the prepush waypoint.  It closes only after
    reaching contact, so there is no strange pause halfway through approach.
    """
    env.reset(
        seed=int(scene_seed),
        options={
            "cube_xy": np.asarray(cube_xy, np.float32).tolist(),
        },
    )

    planner = new_planner(env, vis=vis)
    e = env.unwrapped

    try:
        cube = get_cube_xyz(env).astype(np.float64)

        tcp_q = np.asarray(
            e.agent.tcp.pose.sp.q,
            dtype=np.float64,
        )

        # -------------------------------------------------------------
        # 1) FIRST MOVEMENT = PUSH INTENTION
        #
        # Go directly to a low point behind the cube.
        # This makes the trajectory diverge from Pick naturally without
        # adding a task-label-like motion prefix.
        # -------------------------------------------------------------
        prepush_center = cube + np.array(
            [-float(prepush_distance), 0.0, 0.0],
            dtype=np.float64,
        )

        prepush_pose = sapien.Pose(
            p=prepush_center,
            q=tcp_q,
        )

        if not planner_step_ok(
            planner.move_to_pose_with_screw(prepush_pose)
        ):
            print("  PUSH failed at stage: low-align-behind-cube")
            return None

        # -------------------------------------------------------------
        # 2) APPROACH CONTACT WHILE STILL OPEN.
        #
        # This removes the old gripper-close pause at the prepush point.
        # -------------------------------------------------------------
        contact_center = cube + np.array(
            [-float(contact_offset), 0.0, 0.0],
            dtype=np.float64,
        )

        contact_pose = sapien.Pose(
            p=contact_center,
            q=tcp_q,
        )

        if not planner_step_ok(
            planner.move_to_pose_with_screw(contact_pose)
        ):
            print("  PUSH failed at stage: approach-contact")
            return None

        # Closing here is intentional: this is the physical interaction point.
        planner.close_gripper()

        # -------------------------------------------------------------
        # 3) ONE CONTINUOUS FORWARD PUSH.
        # -------------------------------------------------------------
        push_center = cube + np.array(
            [
                float(push_tcp_travel) - float(contact_offset),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        push_pose = sapien.Pose(
            p=push_center,
            q=tcp_q,
        )

        if not planner_step_ok(
            planner.move_to_pose_with_screw(push_pose)
        ):
            print("  PUSH failed at stage: continuous-forward-push")
            return None

        return env.episode()

    finally:
        planner.close()


# =====================================================================
# Physical evaluation of demonstrations
# =====================================================================

def evaluate_episode(
    kind,
    ep,
    pick_lift,
    push_distance,
    pick_tolerance,
    push_tolerance,
):
    initial = ep["cube_xyz"][0].astype(np.float64)
    final = ep["cube_xyz"][-1].astype(np.float64)

    if kind == "pick":
        target = initial + np.array(
            [0.0, 0.0, float(pick_lift)],
            dtype=np.float64,
        )

        target_error = float(
            np.linalg.norm(final - target)
        )

        lift = float(
            final[2] - initial[2]
        )

        xy_drift = float(
            np.linalg.norm(
                final[:2] - initial[:2]
            )
        )

        success = (
            target_error <= float(pick_tolerance)
            and lift >= float(pick_lift - pick_tolerance)
        )

        return {
            "target": target.astype(np.float32),
            "success": bool(success),
            "target_error": target_error,
            "primary_displacement": lift,
            "lateral_error": xy_drift,
            "z_error": abs(float(final[2] - target[2])),
        }

    target = initial + np.array(
        [float(push_distance), 0.0, 0.0],
        dtype=np.float64,
    )

    xy_error = float(
        np.linalg.norm(
            final[:2] - target[:2]
        )
    )

    dx = float(
        final[0] - initial[0]
    )

    dy = float(
        final[1] - initial[1]
    )

    dz = float(
        final[2] - initial[2]
    )

    success = (
        xy_error <= float(push_tolerance)
        and abs(dz) <= 0.012
        and dx >= float(push_distance - push_tolerance)
    )

    return {
        "target": target.astype(np.float32),
        "success": bool(success),
        "target_error": xy_error,
        "primary_displacement": dx,
        "lateral_error": abs(dy),
        "z_error": abs(dz),
    }


def initial_pair_check(pick, push):
    rgb_mae = float(
        np.mean(
            np.abs(
                pick["rgb"][0].astype(np.float32)
                - push["rgb"][0].astype(np.float32)
            )
        )
    )

    q_error = float(
        np.max(
            np.abs(
                pick["qpos"][0]
                - push["qpos"][0]
            )
        )
    )

    cube_error = float(
        np.max(
            np.abs(
                pick["cube_xyz"][0]
                - push["cube_xyz"][0]
            )
        )
    )

    if rgb_mae > 0.25:
        raise RuntimeError(
            f"Paired initial RGB differs: MAE={rgb_mae:.6f}. "
            "Possible task leakage."
        )

    if q_error > 1e-5:
        raise RuntimeError(
            f"Paired initial qpos differs: max={q_error:.8f}"
        )

    if cube_error > 1e-6:
        raise RuntimeError(
            f"Paired initial cube pose differs: max={cube_error:.8f}"
        )

    return {
        "rgb_mae": rgb_mae,
        "qmax": q_error,
        "cubemax": cube_error,
    }


def early_divergence_frame(pick, push, threshold=0.01):
    n = min(
        len(pick["tcp_xyz"]),
        len(push["tcp_xyz"]),
    )

    d = np.linalg.norm(
        pick["tcp_xyz"][:n]
        - push["tcp_xyz"][:n],
        axis=1,
    )

    idx = np.flatnonzero(
        d >= float(threshold)
    )

    return (
        int(idx[0])
        if len(idx)
        else -1
    )


# =====================================================================
# Split + storage
# =====================================================================

def make_stratified_grid(
    count,
    grid_side,
    x_min,
    x_max,
    y_min,
    y_max,
):
    """
    Divide the practical rectangular workspace into grid cells.

    Recommended final configuration:
        grid_side = 10
        count = 100

    Workspace defaults are intentionally asymmetric because the measured
    practical Push workspace is more restricted toward +X:

        X = [-0.20, +0.10] m
        Y = [-0.15, +0.15] m

    This is a 30 cm x 30 cm square.
    """
    count = int(count)
    grid_side = int(grid_side)

    x_min = float(x_min)
    x_max = float(x_max)
    y_min = float(y_min)
    y_max = float(y_max)

    if grid_side < 1:
        raise ValueError("grid_side must be >= 1")

    if not x_min < x_max:
        raise ValueError("x_min must be < x_max")

    if not y_min < y_max:
        raise ValueError("y_min must be < y_max")

    total = grid_side * grid_side

    if count > total:
        raise ValueError(
            f"count={count} exceeds grid capacity "
            f"{grid_side}x{grid_side}={total}"
        )

    x_edges = np.linspace(
        x_min,
        x_max,
        grid_side + 1,
        dtype=np.float64,
    )

    y_edges = np.linspace(
        y_min,
        y_max,
        grid_side + 1,
        dtype=np.float64,
    )

    cells = []

    for row in range(grid_side):
        for col in range(grid_side):
            x0 = float(x_edges[col])
            x1 = float(x_edges[col + 1])
            y0 = float(y_edges[row])
            y1 = float(y_edges[row + 1])

            cells.append(
                {
                    "grid_row": row,
                    "grid_col": col,
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "center_x": 0.5 * (x0 + x1),
                    "center_y": 0.5 * (y0 + y1),
                }
            )

    if count == total:
        return cells

    # Smoke tests sample cells spread over the COMPLETE workspace.
    chosen = np.rint(
        np.linspace(
            0,
            total - 1,
            count,
        )
    ).astype(int)

    return [cells[int(i)] for i in chosen]


def sample_xy_in_cell(cell, rng, jitter_fraction):
    """
    Sample a position around the cell center while staying within the cell.

    jitter_fraction:
        0.0 -> exact center
        0.6 -> up to 60% of center-to-edge distance (recommended)
        1.0 -> can reach the cell edge
    """
    jf = float(jitter_fraction)

    if not 0.0 <= jf <= 1.0:
        raise ValueError("grid jitter must be between 0 and 1")

    hx = 0.5 * (float(cell["x1"]) - float(cell["x0"]))
    hy = 0.5 * (float(cell["y1"]) - float(cell["y0"]))

    return np.asarray(
        [
            float(cell["center_x"]) + rng.uniform(-hx * jf, +hx * jf),
            float(cell["center_y"]) + rng.uniform(-hy * jf, +hy * jf),
        ],
        dtype=np.float32,
    )


def split_ids(count, seed, grid_side=10):
    """
    Spatially balanced scene split.

    For the definitive 10x10 / 100-scene dataset:
      - every ROW has 8 train, 1 validation, 1 test cell
      - every COLUMN has exactly 1 validation and 1 test cell

    Pick/Push from the same scene always remain in the same split.

    For smoke tests or non-full grids, fall back to deterministic random
    scene-pair splitting.
    """
    count = int(count)
    grid_side = int(grid_side)

    if count == grid_side * grid_side and grid_side == 10:
        split = {}

        # Because gcd(3,10)=1, these visit every column exactly once.
        for row in range(grid_side):
            test_col = (3 * row + 1) % grid_side
            val_col = (3 * row + 6) % grid_side

            for col in range(grid_side):
                scene_id = row * grid_side + col

                if col == test_col:
                    split[scene_id] = "test"
                elif col == val_col:
                    split[scene_id] = "validation"
                else:
                    split[scene_id] = "train"

        return split

    rng = np.random.default_rng(seed)
    ids = rng.permutation(count)

    n_train = int(round(count * 0.8))
    n_val = int(round(count * 0.1))

    return (
        {int(x): "train" for x in ids[:n_train]}
        | {
            int(x): "validation"
            for x in ids[n_train:n_train + n_val]
        }
        | {
            int(x): "test"
            for x in ids[n_train + n_val:]
        }
    )


def write_episode(
    target_h5,
    group_name,
    scene_id,
    kind,
    split,
    ep,
    metrics,
    agent_ids,
    interaction_ids,
    offline_frames,
    pair_check,
    divergence_frame,
    grid_row,
    grid_col,
    grid_center_x,
    grid_center_y,
):
    label = PICK_LABEL if kind == "pick" else PUSH_LABEL

    agent_mask, interaction_mask = make_masks(
        ep["segmentation"],
        agent_ids,
        interaction_ids,
    )

    g = target_h5.create_group(
        f"trajectories/{group_name}"
    )

    g.attrs["task"] = ENV_ID
    g.attrs["kind"] = kind
    g.attrs["label"] = label
    g.attrs["split"] = split
    g.attrs["scene_id"] = int(scene_id)
    g.attrs["grid_row"] = int(grid_row)
    g.attrs["grid_col"] = int(grid_col)
    g.attrs["grid_center_x"] = float(grid_center_x)
    g.attrs["grid_center_y"] = float(grid_center_y)
    g.attrs["paired_scene"] = True
    g.attrs["camera"] = "diagonal_right"
    g.attrs["goal_visible"] = False
    g.attrs["goal_is_model_input"] = False
    g.attrs["interaction_source_semantics"] = "cube"
    g.attrs["agent_source_semantics"] = "Panda robot"
    g.attrs["interaction_segmentation_ids"] = interaction_ids
    g.attrs["agent_segmentation_ids"] = agent_ids
    g.attrs["fixed_behavior_target"] = True
    g.attrs["success_at_end"] = bool(metrics["success"])
    g.attrs["success_ever"] = bool(metrics["success"])
    g.attrs["target_error"] = float(metrics["target_error"])
    g.attrs["primary_displacement"] = float(
        metrics["primary_displacement"]
    )
    g.attrs["lateral_error"] = float(
        metrics["lateral_error"]
    )
    g.attrs["z_error"] = float(
        metrics["z_error"]
    )
    g.attrs["pair_initial_rgb_mae"] = float(
        pair_check["rgb_mae"]
    )
    g.attrs["pair_initial_qpos_max_error"] = float(
        pair_check["qmax"]
    )
    g.attrs["pair_initial_cube_max_error"] = float(
        pair_check["cubemax"]
    )
    g.attrs["pair_tcp_divergence_frame_1cm"] = int(
        divergence_frame
    )

    g.create_dataset(
        "rgb",
        data=ep["rgb"],
        compression="gzip",
        compression_opts=1,
        chunks=(1, *ep["rgb"].shape[1:]),
    )

    g.create_dataset(
        "agent_mask",
        data=agent_mask,
        compression="gzip",
        compression_opts=1,
        chunks=(1, *agent_mask.shape[1:]),
    )

    g.create_dataset(
        "interaction_mask",
        data=interaction_mask,
        compression="gzip",
        compression_opts=1,
        chunks=(1, *interaction_mask.shape[1:]),
    )

    g.create_dataset(
        "qpos",
        data=ep["qpos"],
        compression="gzip",
        compression_opts=1,
    )

    # Compatibility with existing data_utils.py.
    # This is an EVALUATION target only; keep USE_GOAL=False.
    g.create_dataset(
        "goal_xyz",
        data=metrics["target"],
    )

    g.create_dataset(
        "evaluation_target_xyz",
        data=metrics["target"],
    )

    g.create_dataset(
        "initial_cube_xyz",
        data=ep["cube_xyz"][0],
    )

    g.create_dataset(
        "cube_xyz",
        data=ep["cube_xyz"],
        compression="gzip",
        compression_opts=1,
    )

    g.create_dataset(
        "tcp_xyz",
        data=ep["tcp_xyz"],
        compression="gzip",
        compression_opts=1,
    )

    g.create_dataset(
        "actions",
        data=ep["actions"],
        compression="gzip",
        compression_opts=1,
    )

    idx = np.rint(
        np.linspace(
            0,
            len(ep["rgb"]) - 1,
            int(offline_frames),
        )
    ).astype(np.int32)

    g.create_dataset(
        "offline_normalized_indices",
        data=idx,
    )

    return {
        "id": group_name,
        "scene_id": int(scene_id),
        "grid_row": int(grid_row),
        "grid_col": int(grid_col),
        "grid_center_x": float(grid_center_x),
        "grid_center_y": float(grid_center_y),
        "kind": kind,
        "label": label,
        "split": split,
        "frames": int(len(ep["rgb"])),
        "success": int(metrics["success"]),
        "target_error": float(metrics["target_error"]),
        "primary_displacement": float(
            metrics["primary_displacement"]
        ),
        "lateral_error": float(metrics["lateral_error"]),
        "z_error": float(metrics["z_error"]),
        "initial_rgb_mae_pair": float(
            pair_check["rgb_mae"]
        ),
        "tcp_divergence_frame_1cm": int(
            divergence_frame
        ),
        "cube_x": float(ep["cube_xyz"][0, 0]),
        "cube_y": float(ep["cube_xyz"][0, 1]),
    }


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
    )

    ap.add_argument(
        "--count",
        type=int,
        default=100,
        help="Accepted paired scenes. Produces 2*count trajectories.",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--max-candidates",
        type=int,
        default=400,
        help="Maximum scene candidates before giving up.",
    )

    ap.add_argument(
        "--x-min",
        type=float,
        default=-0.20,
        help="Minimum initial cube X [m]. Default -0.20.",
    )

    ap.add_argument(
        "--x-max",
        type=float,
        default=0.10,
        help="Maximum initial cube X [m]. Default +0.10.",
    )

    ap.add_argument(
        "--y-min",
        type=float,
        default=-0.15,
        help="Minimum initial cube Y [m]. Default -0.15.",
    )

    ap.add_argument(
        "--y-max",
        type=float,
        default=0.15,
        help="Maximum initial cube Y [m]. Default +0.15.",
    )

    ap.add_argument(
        "--grid-side",
        type=int,
        default=10,
        help="Grid cells per axis. 10 -> 100 spatial cells.",
    )

    ap.add_argument(
        "--grid-jitter",
        type=float,
        default=0.60,
        help=(
            "Within-cell jitter fraction [0,1]. "
            "Default 0.60 keeps samples away from exact grid centers."
        ),
    )

    ap.add_argument(
        "--attempts-per-cell",
        type=int,
        default=8,
        help=(
            "Retries inside the SAME spatial cell if motion planning fails."
        ),
    )

    ap.add_argument(
        "--pick-lift",
        type=float,
        default=0.10,
        help="Fixed Pick displacement in +Z [m].",
    )

    ap.add_argument(
        "--push-distance",
        type=float,
        default=0.10,
        help="Desired fixed CUBE displacement in +X [m].",
    )

    ap.add_argument(
        "--push-tcp-travel",
        type=float,
        default=0.13,
        help=(
            "Commanded TCP forward travel [m]. "
            "Calibrated from smoke tests so the cube moves about 0.10 m."
        ),
    )

    ap.add_argument(
        "--pick-tolerance",
        type=float,
        default=0.015,
        help="Accepted Pick final-target error [m].",
    )

    ap.add_argument(
        "--push-tolerance",
        type=float,
        default=0.015,
        help="Accepted Push final XY target error [m].",
    )

    ap.add_argument(
        "--pregrasp-height",
        type=float,
        default=0.05,
        help="Pick first waypoint above the cube [m].",
    )

    ap.add_argument(
        "--prepush-distance",
        type=float,
        default=0.08,
        help="Push first waypoint behind cube [m].",
    )

    ap.add_argument(
        "--push-contact-offset",
        type=float,
        default=0.05,
        help="TCP X offset behind cube center at contact [m].",
    )

    ap.add_argument(
        "--push-tcp-z",
        type=float,
        default=0.035,
        help="Horizontal push TCP height above table [m].",
    )

    ap.add_argument(
        "--offline-frames",
        type=int,
        default=80,
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "datasets/maniskill_intent_v1"
        ),
    )

    ap.add_argument(
        "--vis",
        action="store_true",
        help="Open motion-planning visualization.",
    )

    args = ap.parse_args()

    if args.count < 1:
        ap.error("--count must be >=1")

    if args.offline_frames < 2:
        ap.error("--offline-frames must be >=2")

    if args.grid_side < 1:
        ap.error("--grid-side must be >=1")

    if not 0.0 <= args.grid_jitter <= 1.0:
        ap.error("--grid-jitter must be between 0 and 1")

    if args.attempts_per_cell < 1:
        ap.error("--attempts-per-cell must be >=1")

    if not args.x_min < args.x_max:
        ap.error("--x-min must be < --x-max")

    if not args.y_min < args.y_max:
        ap.error("--y-min must be < --y-max")

    if args.count > args.grid_side * args.grid_side:
        ap.error(
            f"--count {args.count} exceeds grid capacity "
            f"{args.grid_side}x{args.grid_side}"
        )

    out = args.out_dir.expanduser().resolve()
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    h5_path = out / "multimodal_intent_v1.h5"
    csv_path = out / "manifest_intent_v1.csv"
    summary_path = out / "summary_intent_v1.json"

    for path in (
        h5_path,
        csv_path,
        summary_path,
    ):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. "
                "Old datasets are never overwritten."
            )

    # One shared environment for both behavior classes.
    base_env = gym.make(
        ENV_ID,
        obs_mode="rgb+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        max_episode_steps=160,
    )

    env = CaptureEnv(base_env)

    # Initialize once so segmentation_id_map exists.
    env.reset(
        seed=args.seed,
        options={
            "cube_xy": [0.0, 0.0],
        },
    )

    (
        agent_ids,
        interaction_ids,
        goal_ids,
    ) = resolve_segmentation_ids(env)

    split_map = split_ids(
        args.count,
        args.seed,
        grid_side=args.grid_side,
    )

    rows = []
    accepted = 0
    rejected = 0

    grid_cells = make_stratified_grid(
        count=args.count,
        grid_side=args.grid_side,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )

    # Independent deterministic RNG for within-cell jitter.
    scene_rng = np.random.default_rng(
        args.seed + 99173
    )

    try:
        with h5py.File(
            h5_path,
            "w",
        ) as target:

            target.attrs.update(
                {
                    "dataset_name":
                        "ManiSkill paired motor-intention Pick/Push v1",
                    "schema_version":
                        "intent_v1_smooth_grid_large_v5",
                    "seed":
                        int(args.seed),
                    "paired_scenes":
                        True,
                    "camera":
                        "diagonal_right",
                    "camera_eye":
                        np.asarray(CAMERA_EYE, np.float32),
                    "camera_target":
                        np.asarray(CAMERA_TARGET, np.float32),
                    "goal_visible":
                        False,
                    "goal_model_input_recommended":
                        False,
                    "interaction_mask_semantics":
                        "cube / generic interaction region",
                    "pick_rule":
                        (
                            f"rise first, align above cube at safe Z, "
                            f"single vertical descent, grasp, fixed +Z "
                            f"{args.pick_lift:.3f} m"
                        ),
                    "push_rule":
                        (
                            f"low alignment behind cube, contact, "
                            f"single forward push; desired cube +X "
                            f"{args.push_distance:.3f} m; "
                            f"TCP travel {args.push_tcp_travel:.3f} m"
                        ),
                    "split_unit":
                        "scene_pair",
                    "spatial_sampling":
                        "stratified_grid_with_within_cell_jitter",
                    "grid_side":
                        int(args.grid_side),
                    "grid_cells_total":
                        int(args.grid_side * args.grid_side),
                    "grid_cells_used":
                        int(args.count),
                    "workspace_x_min_m":
                        float(args.x_min),
                    "workspace_x_max_m":
                        float(args.x_max),
                    "workspace_y_min_m":
                        float(args.y_min),
                    "workspace_y_max_m":
                        float(args.y_max),
                    "workspace_width_x_m":
                        float(args.x_max - args.x_min),
                    "workspace_width_y_m":
                        float(args.y_max - args.y_min),
                    "split_method":
                        "spatially_balanced_rows_and_columns",
                    "grid_jitter_fraction":
                        float(args.grid_jitter),
                    "attempts_per_cell":
                        int(args.attempts_per_cell),
                    "offline_frames":
                        int(args.offline_frames),
                }
            )

            candidate = 0

            for scene_id, cell in enumerate(grid_cells):
                split = split_map[scene_id]
                accepted_this_cell = False

                print()
                print("=" * 72)
                print(
                    f"GRID scene={scene_id:03d} "
                    f"cell=({cell['grid_row']},{cell['grid_col']}) "
                    f"center=({cell['center_x']:+.4f},"
                    f"{cell['center_y']:+.4f}) "
                    f"split={split}"
                )
                print("=" * 72)

                for cell_attempt in range(
                    int(args.attempts_per_cell)
                ):
                    if candidate >= int(args.max_candidates):
                        raise RuntimeError(
                            f"Reached --max-candidates={args.max_candidates} "
                            f"while generating cell "
                            f"({cell['grid_row']},{cell['grid_col']})."
                        )

                    cube_xy = sample_xy_in_cell(
                        cell,
                        scene_rng,
                        args.grid_jitter,
                    )

                    # Deterministic seed per scene/retry.
                    scene_seed = int(
                        args.seed
                        + scene_id * 1000
                        + cell_attempt
                    )

                    print(
                        f"  attempt={cell_attempt + 1}/"
                        f"{args.attempts_per_cell} "
                        f"global_candidate={candidate:03d} "
                        f"cube_xy={np.round(cube_xy, 4)}"
                    )

                    candidate += 1

                    try:
                        pick = run_pick(
                            env,
                            scene_seed=scene_seed,
                            cube_xy=cube_xy,
                            lift_distance=args.pick_lift,
                            pregrasp_height=args.pregrasp_height,
                            vis=args.vis,
                        )

                        push = run_push(
                            env,
                            scene_seed=scene_seed,
                            cube_xy=cube_xy,
                            push_distance=args.push_distance,
                            push_tcp_travel=args.push_tcp_travel,
                            prepush_distance=args.prepush_distance,
                            contact_offset=args.push_contact_offset,
                            push_tcp_z=args.push_tcp_z,
                            vis=args.vis,
                        )

                    except Exception as exc:
                        print(
                            "    motion-planning error:",
                            repr(exc),
                        )
                        rejected += 1
                        continue

                    if pick is None or push is None:
                        print(
                            "    rejected: motion planner failed"
                        )
                        rejected += 1
                        continue

                    assert_goal_not_visible(
                        pick["segmentation"],
                        goal_ids,
                    )
                    assert_goal_not_visible(
                        push["segmentation"],
                        goal_ids,
                    )

                    pair_check = initial_pair_check(
                        pick,
                        push,
                    )

                    pick_metrics = evaluate_episode(
                        "pick",
                        pick,
                        pick_lift=args.pick_lift,
                        push_distance=args.push_distance,
                        pick_tolerance=args.pick_tolerance,
                        push_tolerance=args.push_tolerance,
                    )

                    push_metrics = evaluate_episode(
                        "push",
                        push,
                        pick_lift=args.pick_lift,
                        push_distance=args.push_distance,
                        pick_tolerance=args.pick_tolerance,
                        push_tolerance=args.push_tolerance,
                    )

                    print(
                        "    Pick:",
                        f"success={pick_metrics['success']}",
                        f"lift={pick_metrics['primary_displacement']:.3f}",
                        f"err={pick_metrics['target_error']:.3f}",
                    )

                    print(
                        "    Push:",
                        f"success={push_metrics['success']}",
                        f"dx={push_metrics['primary_displacement']:.3f}",
                        f"err={push_metrics['target_error']:.3f}",
                        f"zerr={push_metrics['z_error']:.3f}",
                    )

                    if (
                        not pick_metrics["success"]
                        or not push_metrics["success"]
                    ):
                        print(
                            "    rejected: pair must succeed in BOTH behaviors"
                        )
                        rejected += 1
                        continue

                    divergence = early_divergence_frame(
                        pick,
                        push,
                        threshold=0.01,
                    )

                    print(
                        "    ACCEPTED",
                        f"scene={scene_id:03d}",
                        f"cell=({cell['grid_row']},{cell['grid_col']})",
                        "initial_rgb_mae=",
                        f"{pair_check['rgb_mae']:.6f}",
                        "tcp_divergence_frame=",
                        divergence,
                    )

                    for kind, ep, metrics in (
                        ("pick", pick, pick_metrics),
                        ("push", push, push_metrics),
                    ):
                        group_name = (
                            f"{kind}_{scene_id:03d}"
                        )

                        rows.append(
                            write_episode(
                                target,
                                group_name=group_name,
                                scene_id=scene_id,
                                kind=kind,
                                split=split,
                                ep=ep,
                                metrics=metrics,
                                agent_ids=agent_ids,
                                interaction_ids=interaction_ids,
                                offline_frames=args.offline_frames,
                                pair_check=pair_check,
                                divergence_frame=divergence,
                                grid_row=cell["grid_row"],
                                grid_col=cell["grid_col"],
                                grid_center_x=cell["center_x"],
                                grid_center_y=cell["center_y"],
                            )
                        )

                    accepted += 1
                    accepted_this_cell = True
                    break

                if not accepted_this_cell:
                    raise RuntimeError(
                        f"Could not generate a valid Pick+Push pair for "
                        f"grid cell ({cell['grid_row']},{cell['grid_col']}) "
                        f"after {args.attempts_per_cell} attempts. "
                        "Aborting instead of leaving a spatial hole."
                    )

            if accepted != args.count:
                raise RuntimeError(
                    f"Expected {args.count} accepted grid scenes, got {accepted}."
                )

        with csv_path.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "accepted_scene_pairs": accepted,
            "trajectories": len(rows),
            "rejected_candidates": rejected,
            "per_class": {
                "pick": sum(
                    r["kind"] == "pick"
                    for r in rows
                ),
                "push": sum(
                    r["kind"] == "push"
                    for r in rows
                ),
            },
            "splits": {
                s: {
                    "scene_pairs": sum(
                        split_map[i] == s
                        for i in range(args.count)
                    ),
                    "trajectories": sum(
                        r["split"] == s
                        for r in rows
                    ),
                }
                for s in (
                    "train",
                    "validation",
                    "test",
                )
            },
            "spatial_grid": {
                "side": int(args.grid_side),
                "total_cells": int(args.grid_side * args.grid_side),
                "used_cells": int(args.count),
                "x_min_m": float(args.x_min),
                "x_max_m": float(args.x_max),
                "y_min_m": float(args.y_min),
                "y_max_m": float(args.y_max),
                "width_x_m": float(args.x_max - args.x_min),
                "width_y_m": float(args.y_max - args.y_min),
                "jitter_fraction": float(args.grid_jitter),
                "split_method": "spatially_balanced_rows_and_columns",
                "attempts_per_cell": int(args.attempts_per_cell),
                "coverage_rule": (
                    "one accepted Pick+Push pair per selected cell; "
                    "failed planners retry inside the same cell"
                ),
            },
            "fixed_targets_m": {
                "pick_plus_z": float(
                    args.pick_lift
                ),
                "push_cube_plus_x": float(
                    args.push_distance
                ),
                "push_tcp_travel": float(
                    args.push_tcp_travel
                ),
            },
            "camera": {
                "name": "diagonal_right",
                "eye": CAMERA_EYE,
                "target": CAMERA_TARGET,
            },
            "anti_leakage": {
                "goal_visible": False,
                "paired_initial_scene": True,
                "split_by_scene_pair": True,
                "mean_initial_rgb_mae": float(
                    np.mean(
                        [
                            r["initial_rgb_mae_pair"]
                            for r in rows[::2]
                        ]
                    )
                ),
                "recommended_K1_classifier_accuracy":
                    "approximately chance (~0.5); investigate leakage if much higher",
            },
            "mean_tcp_divergence_frame_1cm": float(
                np.mean(
                    [
                        r["tcp_divergence_frame_1cm"]
                        for r in rows[::2]
                        if r["tcp_divergence_frame_1cm"] >= 0
                    ]
                )
            ),
            "notes": [
                "Old datasets are untouched.",
                "Cube positions use a 30x30 cm asymmetric practical workspace: X[-20,+10] cm, Y[-15,+15] cm.",
                "Workspace bounds come from the measured Pick+Push practical-workspace probe.",
                "Cube positions use a stratified spatial grid with within-cell jitter.",
                "Full 10x10 split is spatially balanced: every row and column contributes validation/test coverage.",
                "Each selected grid cell contributes exactly one accepted paired scene.",
                "Both classes interact with the same cube and same interaction_mask semantics.",
                "Class intent must emerge from motor behavior, not a visible target.",
                "Pick never descends until XY-aligned above the grasp point.",
                "Pick uses one continuous final descent to remove mid-descent pause.",
                "Push first movement is naturally low/behind; no artificial cue waypoint.",
                "Push closes the gripper only at contact, removing the old pre-contact pause.",
                "goal_xyz is evaluation-only compatibility metadata; keep USE_GOAL=False.",
                "Pick final objective is fixed vertical lift.",
                "Push final objective is fixed +X displacement with no visible bullseye.",
            ],
        }

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("\nCOMPLETE")
        print("H5      :", h5_path)
        print("Manifest:", csv_path)
        print("Summary :", summary_path)
        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

    except BaseException:
        # Avoid leaving a corrupted/partial HDF5, including Ctrl-C.
        h5_path.unlink(
            missing_ok=True
        )
        raise

    finally:
        env.close()


if __name__ == "__main__":
    main()