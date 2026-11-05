#!/usr/bin/env python3
"""
Build a generalized object-structured ManiSkill dataset for PickCube/PushCube.

The model-facing semantics are intentionally generic:

    agent_mask        = pixels belonging to the acting body/manipulator
    interaction_mask  = pixels belonging to the object/region being interacted with

For the current ManiSkill experiments:
    agent_mask        -> Panda robot links
    interaction_mask  -> cube

This lets the same downstream ESN interface be reused later for other domains, e.g.
a robot arm approaching a pressure sensor:

    agent_mask        -> arm
    interaction_mask  -> pressure sensor / sensitive region

RGB, qpos, goal_xyz and cube_xyz remain synchronized at native replay frame indices.
Masks are stored at native camera resolution; resize them later with nearest-neighbor.
"""

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import mani_skill.envs  # registers ManiSkill environments


# ---------------------------------------------------------------------
# Task-specific mapping layer.
# Everything written to the output dataset uses generic mask names.
# ---------------------------------------------------------------------

TASKS = {
    "pick": {
        "env_id": "PickCube-v1",
        "goal_actor": "goal_site",
        "agent_name_tokens": ("panda",),
        "interaction_names": ("cube",),
    },
    "push": {
        "env_id": "PushCube-v1",
        "goal_actor": "goal_region",
        "agent_name_tokens": ("panda",),
        "interaction_names": ("cube",),
    },
}

RGB_KEY = "obs/sensor_data/base_camera/rgb"
SEG_KEY = "obs/sensor_data/base_camera/segmentation"
QPOS_KEY = "obs/agent/qpos"


def ordered_trajectory_keys(f):
    return sorted(
        (
            k
            for k in f
            if k.startswith("traj_") and k.split("_")[-1].isdigit()
        ),
        key=lambda k: int(k.split("_")[-1]),
    )


def find_segmentation_replay(folder):
    """
    Find exactly one replay HDF5 that contains RGB + segmentation + qpos.
    We inspect content rather than trust the filename.
    """
    candidates = []

    for path in sorted(folder.glob("trajectory*.h5")):
        if path.name == "trajectory.h5":
            continue

        try:
            with h5py.File(path, "r") as f:
                keys = ordered_trajectory_keys(f)
                if not keys:
                    continue

                sample = f[keys[0]]
                if (
                    RGB_KEY in sample
                    and SEG_KEY in sample
                    and QPOS_KEY in sample
                ):
                    candidates.append(path)
        except OSError:
            continue

    if not candidates:
        raise RuntimeError(
            f"No RGB+segmentation replay found in {folder}.\n"
            "Generate one with ManiSkill replay_trajectory using "
            '-o "rgb+segmentation" --use-env-states --save-traj.'
        )

    if len(candidates) > 1:
        names = "\n".join(f"  {p.name}" for p in candidates)
        raise RuntimeError(
            f"Multiple RGB+segmentation replays found in {folder}:\n{names}\n"
            "Move old duplicates away so the source is unambiguous."
        )

    return candidates[0]


def split_ids(count, seed):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(count)
    n_train = int(round(count * 0.8))
    n_val = int(round(count * 0.1))

    return (
        {int(x): "train" for x in ids[:n_train]}
        | {int(x): "validation" for x in ids[n_train:n_train + n_val]}
        | {int(x): "test" for x in ids[n_train + n_val:]}
    )


def get_segmentation_ids(env_id, agent_name_tokens, interaction_names):
    """
    Instantiate the same task and resolve scene segmentation IDs.

    The result is validated later against the IDs actually present in the replay.
    """
    env = gym.make(env_id, obs_mode="rgb+segmentation")

    try:
        env.reset(seed=0)
        mapping = env.unwrapped.segmentation_id_map

        agent_ids = []
        interaction_ids = []

        agent_tokens = tuple(x.lower() for x in agent_name_tokens)
        interaction_names = {x.lower() for x in interaction_names}

        print(f"\nSEGMENTATION MAP: {env_id}")

        for obj_id, obj in sorted(mapping.items(), key=lambda x: int(x[0])):
            obj_id = int(obj_id)
            name = str(getattr(obj, "name", ""))
            lname = name.lower()

            print(
                f"  ID={obj_id:3d} "
                f"type={type(obj).__name__:15s} "
                f"name={name}"
            )

            if any(token in lname for token in agent_tokens):
                agent_ids.append(obj_id)

            if lname in interaction_names:
                interaction_ids.append(obj_id)

        if not agent_ids:
            raise RuntimeError(
                f"{env_id}: no agent segmentation IDs matched "
                f"tokens={agent_name_tokens}"
            )

        if not interaction_ids:
            raise RuntimeError(
                f"{env_id}: no interaction-object IDs matched "
                f"names={tuple(interaction_names)}"
            )

        print(f"  agent IDs       = {agent_ids}")
        print(f"  interaction IDs = {interaction_ids}")

        return (
            np.asarray(agent_ids, dtype=np.int32),
            np.asarray(interaction_ids, dtype=np.int32),
        )

    finally:
        env.close()


def canonicalize_segmentation(segmentation):
    seg = np.asarray(segmentation)

    if seg.ndim == 4 and seg.shape[-1] == 1:
        seg = seg[..., 0]

    if seg.ndim != 3:
        raise ValueError(
            "Expected segmentation shape (T,H,W) or (T,H,W,1), "
            f"got {seg.shape}"
        )

    return seg


def segmentation_to_generic_masks(segmentation, agent_ids, interaction_ids):
    seg = canonicalize_segmentation(segmentation)

    agent_mask = np.isin(seg, agent_ids).astype(np.uint8)
    interaction_mask = np.isin(seg, interaction_ids).astype(np.uint8)

    return agent_mask, interaction_mask


def validate_masks(
    env_id,
    key,
    segmentation,
    agent_ids,
    interaction_ids,
    agent_mask,
    interaction_mask,
):
    seg = canonicalize_segmentation(segmentation)
    unique_ids = np.unique(seg).astype(np.int64)

    if np.intersect1d(unique_ids, agent_ids).size == 0:
        raise ValueError(
            f"{env_id}/{key}: expected agent IDs {agent_ids.tolist()} "
            f"not present in replay segmentation. "
            f"Observed IDs={unique_ids.tolist()}"
        )

    if np.intersect1d(unique_ids, interaction_ids).size == 0:
        raise ValueError(
            f"{env_id}/{key}: expected interaction IDs "
            f"{interaction_ids.tolist()} not present in replay segmentation. "
            f"Observed IDs={unique_ids.tolist()}"
        )

    if int(agent_mask.sum()) == 0:
        raise ValueError(f"{env_id}/{key}: agent_mask is empty")

    if int(interaction_mask.sum()) == 0:
        raise ValueError(f"{env_id}/{key}: interaction_mask is empty")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of trajectories per task (default: 100)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--offline-frames",
        type=int,
        default=80,
        help="Store retrospective sampling indices; videos are not resampled",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets/maniskill_multimodal"),
    )
    ap.add_argument(
        "--demo-root",
        type=Path,
        default=Path.home() / ".maniskill" / "demos",
    )
    args = ap.parse_args()

    if args.count < 1 or args.offline_frames < 2:
        ap.error("--count must be positive and --offline-frames at least 2")

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # New files: never overwrite the RGB-only baseline.
    h5_path = out / "multimodal_object_context.h5"
    csv_path = out / "manifest_object_context.csv"
    summary_path = out / "summary_object_context.json"

    if h5_path.exists() or csv_path.exists() or summary_path.exists():
        raise FileExistsError(
            "Object-context outputs already exist. Move/delete them first "
            "to prevent accidental overwrite."
        )

    sources = {}
    segmentation_ids = {}

    # -----------------------------------------------------------------
    # Resolve source files and semantic segmentation IDs.
    # -----------------------------------------------------------------
    for task_key, cfg in TASKS.items():
        env_id = cfg["env_id"]
        folder = args.demo_root / env_id / "motionplanning"
        original = folder / "trajectory.h5"

        if not original.exists():
            raise FileNotFoundError(original)

        replay = find_segmentation_replay(folder)

        with h5py.File(replay, "r") as f:
            keys = ordered_trajectory_keys(f)
            if len(keys) < args.count:
                raise RuntimeError(
                    f"{env_id} has {len(keys)} RGB+segmentation trajectories; "
                    f"need {args.count}"
                )

        sources[task_key] = (original, replay)
        segmentation_ids[task_key] = get_segmentation_ids(
            env_id,
            cfg["agent_name_tokens"],
            cfg["interaction_names"],
        )

        print(f"{env_id}: replay={replay.name}")

    rows = []

    try:
        with h5py.File(h5_path, "w") as target:
            target.attrs.update(
                {
                    "dataset_name":
                        "ManiSkill PickCube+PushCube generalized object-context",
                    "schema_version": "object_context_v1",
                    "seed": args.seed,
                    "episodes_per_task": args.count,
                    "rgb_origin":
                        "Official ManiSkill use-env-states RGB+segmentation replay",
                    "mask_semantics":
                        "agent_mask=acting body; interaction_mask=interaction object/region",
                    "alignment":
                        "Original native frame indices, no frame resampling",
                    "offline_frames": args.offline_frames,
                    "mask_encoding": "binary uint8",
                }
            )

            for task_key, cfg in TASKS.items():
                env_id = cfg["env_id"]
                goal_actor = cfg["goal_actor"]

                original_path, replay_path = sources[task_key]
                agent_ids, interaction_ids = segmentation_ids[task_key]
                assignment = split_ids(args.count, args.seed)

                with (
                    h5py.File(original_path, "r") as original,
                    h5py.File(replay_path, "r") as replay,
                ):
                    keys = ordered_trajectory_keys(replay)[:args.count]

                    for i, key in enumerate(keys):
                        if key not in original:
                            raise KeyError(
                                f"Original trajectory missing: {env_id}/{key}"
                            )

                        observed = replay[key]
                        raw = original[key]

                        rgb = observed[RGB_KEY][:]
                        segmentation = observed[SEG_KEY][:]
                        qpos = observed[QPOS_KEY][:]

                        goal_states = raw[f"env_states/actors/{goal_actor}"][:]
                        goal = goal_states[0, :3].astype(np.float32)
                        cube = raw["env_states/actors/cube"][:, :3].astype(np.float32)

                        steps = len(rgb)

                        if (
                            rgb.ndim != 4
                            or rgb.shape[-1] != 3
                            or rgb.dtype != np.uint8
                        ):
                            raise ValueError(
                                f"{env_id}/{key}: unexpected RGB "
                                f"{rgb.shape} {rgb.dtype}"
                            )

                        seg = canonicalize_segmentation(segmentation)

                        if len(seg) != steps:
                            raise ValueError(
                                f"{env_id}/{key}: segmentation/RGB length mismatch"
                            )

                        if seg.shape[1:3] != rgb.shape[1:3]:
                            raise ValueError(
                                f"{env_id}/{key}: RGB/segmentation spatial mismatch "
                                f"{rgb.shape} / {seg.shape}"
                            )

                        if (
                            qpos.ndim != 2
                            or qpos.shape[1] != 9
                            or len(qpos) != steps
                        ):
                            raise ValueError(
                                f"{env_id}/{key}: unexpected qpos {qpos.shape}"
                            )

                        if len(goal_states) != steps or len(cube) != steps:
                            raise ValueError(
                                f"{env_id}/{key}: state/RGB length mismatch"
                            )

                        if not np.allclose(goal_states[:, :3], goal, atol=1e-5):
                            raise ValueError(
                                f"{env_id}/{key}: goal varies within episode"
                            )

                        if not np.all(np.isfinite(qpos)):
                            raise ValueError(
                                f"{env_id}/{key}: qpos contains non-finite values"
                            )

                        agent_mask, interaction_mask = (
                            segmentation_to_generic_masks(
                                seg,
                                agent_ids,
                                interaction_ids,
                            )
                        )

                        validate_masks(
                            env_id,
                            key,
                            seg,
                            agent_ids,
                            interaction_ids,
                            agent_mask,
                            interaction_mask,
                        )

                        agent_pixels = agent_mask.reshape(steps, -1).sum(axis=1)
                        interaction_pixels = (
                            interaction_mask.reshape(steps, -1).sum(axis=1)
                        )

                        group_name = f"{task_key}_{i:03d}"
                        g = target.create_group(
                            f"trajectories/{group_name}"
                        )

                        g.attrs["task"] = env_id
                        g.attrs["kind"] = task_key
                        g.attrs["label"] = 0 if task_key == "pick" else 1
                        g.attrs["split"] = assignment[i]
                        g.attrs["original_key"] = key
                        g.attrs["source_rgb_segmentation"] = str(replay_path)
                        g.attrs["source_states"] = str(original_path)

                        # Source semantics are metadata only.
                        g.attrs["agent_source_semantics"] = "Panda robot"
                        g.attrs["interaction_source_semantics"] = "cube"
                        g.attrs["agent_segmentation_ids"] = agent_ids
                        g.attrs["interaction_segmentation_ids"] = interaction_ids

                        g.create_dataset(
                            "rgb",
                            data=rgb,
                            compression="gzip",
                            compression_opts=1,
                            chunks=(1, *rgb.shape[1:]),
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
                            data=qpos.astype(np.float32),
                            compression="gzip",
                            compression_opts=1,
                        )

                        g.create_dataset("goal_xyz", data=goal)

                        g.create_dataset(
                            "cube_xyz",
                            data=cube,
                            compression="gzip",
                            compression_opts=1,
                        )

                        offline_indices = np.rint(
                            np.linspace(0, steps - 1, args.offline_frames)
                        ).astype(np.int32)

                        g.create_dataset(
                            "offline_normalized_indices",
                            data=offline_indices,
                        )

                        success = raw["success"][:].astype(bool)
                        g.attrs["success_ever"] = bool(success.any())
                        g.attrs["success_at_end"] = (
                            bool(success[-1]) if len(success) else False
                        )

                        rows.append(
                            {
                                "id": group_name,
                                "task": env_id,
                                "kind": task_key,
                                "label": int(g.attrs["label"]),
                                "split": assignment[i],
                                "original_key": key,
                                "frames": steps,
                                "goal_x": float(goal[0]),
                                "goal_y": float(goal[1]),
                                "goal_z": float(goal[2]),
                                "agent_visible_frames":
                                    int(np.count_nonzero(agent_pixels > 0)),
                                "interaction_visible_frames":
                                    int(np.count_nonzero(interaction_pixels > 0)),
                                "agent_mean_pixels":
                                    float(agent_pixels.mean()),
                                "interaction_mean_pixels":
                                    float(interaction_pixels.mean()),
                                "success_ever":
                                    int(g.attrs["success_ever"]),
                                "success_at_end":
                                    int(g.attrs["success_at_end"]),
                            }
                        )

                        if (i + 1) % 10 == 0 or i == 0:
                            print(
                                f"  {env_id}: {i + 1}/{args.count} "
                                f"| {key} frames={steps} "
                                f"| split={assignment[i]} "
                                f"| agent_px={agent_pixels.mean():.1f} "
                                f"| interaction_px={interaction_pixels.mean():.1f}"
                            )

        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "total": len(rows),
            "per_task": args.count,
            "schema_version": "object_context_v1",
            "splits": {
                s: sum(r["split"] == s for r in rows)
                for s in ("train", "validation", "test")
            },
            "native_frame_range": [
                min(r["frames"] for r in rows),
                max(r["frames"] for r in rows),
            ],
            "successful_once": sum(r["success_ever"] for r in rows),
            "successful_at_end": sum(r["success_at_end"] for r in rows),
            "mean_agent_pixels": float(
                np.mean([r["agent_mean_pixels"] for r in rows])
            ),
            "mean_interaction_pixels": float(
                np.mean([r["interaction_mean_pixels"] for r in rows])
            ),
            "notes": [
                "Generic downstream semantics: agent_mask and interaction_mask",
                "Current ManiSkill mapping: Panda -> agent, cube -> interaction object",
                "Masks are native-resolution binary uint8 and synchronized with RGB",
                "Resize masks only with nearest-neighbor",
                "Masks are supervision targets in the first ESN experiment, not required inputs",
                "The same schema can later represent arm -> agent and pressure sensor -> interaction region",
            ],
        }

        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print(
            "\nCOMPLETE:", h5_path,
            "\nMANIFEST:", csv_path,
            "\nSUMMARY:", summary_path,
            "\n", json.dumps(summary, indent=2),
        )

    except Exception:
        h5_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()