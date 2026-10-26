"""Generalized ManiSkill PickCube+PushCube dataset loader.

Loads synchronized original-length:
    - RGB
    - qpos
    - goal XYZ
    - cube XYZ
    - agent_mask
    - interaction_mask

Generic mask semantics:
    agent_mask:
        acting body / manipulator
        (Panda in current ManiSkill dataset)

    interaction_mask:
        object or sensitive interaction region
        (cube in current ManiSkill dataset)

Masks remain at their native HxW resolution here. The ESN is responsible for
nearest-neighbor resizing to MASK_HEIGHT x MASK_WIDTH.

No temporal resampling.
No data leakage through split normalization.
"""

from pathlib import Path

import h5py
import numpy as np
from PIL import Image

import parameters as p


# ============================================================
# IMAGE RESAMPLING
# ============================================================

RGB_RESAMPLE = getattr(
    Image,
    "Resampling",
    Image,
).LANCZOS


# ============================================================
# DATA PATH
# ============================================================

# Prefer the path declared in parameters.py.
# Keep a sensible fallback so this module can still be inspected/imported
# independently.
DEFAULT_PATH = Path(
    getattr(
        p,
        "DATA_PATH",
        (
            Path(__file__).resolve().parent
            / "datasets"
            / "maniskill_multimodal"
            / "multimodal_object_context.h5"
        ),
    )
)


# ============================================================
# HELPERS
# ============================================================

def _attr_text(value):
    return (
        value.decode("utf-8")
        if isinstance(value, bytes)
        else str(value)
    )


def _process_rgb(rgb):
    """
    Convert native uint8 RGB frames to flattened float32 model input.

    Output:
        RGB mode:
            (T, IMAGE_HEIGHT * IMAGE_WIDTH * 3)

        grayscale mode:
            (T, IMAGE_HEIGHT * IMAGE_WIDTH)
    """
    mode = getattr(
        p,
        "IMAGE_MODE",
        "rgb",
    )

    if mode not in (
        "rgb",
        "grayscale",
    ):
        raise ValueError(
            "IMAGE_MODE must be rgb or grayscale"
        )

    h = int(p.IMAGE_HEIGHT)
    w = int(p.IMAGE_WIDTH)

    channels = (
        3
        if mode == "rgb"
        else 1
    )

    features = (
        h
        * w
        * channels
    )

    frames = np.empty(
        (
            len(rgb),
            features,
        ),
        dtype=np.float32,
    )

    for index, frame in enumerate(rgb):
        im = Image.fromarray(
            frame,
            "RGB",
        )

        im = im.convert(
            "RGB"
            if mode == "rgb"
            else "L"
        )

        im = im.resize(
            (w, h),
            RGB_RESAMPLE,
        )

        frames[index] = (
            np.asarray(
                im,
                dtype=np.float32,
            ).reshape(-1)
            / 255.0
        )

    return frames


def _read_binary_mask(group, name, length, spatial_shape):
    """
    Read one generic binary mask sequence.

    Expected stored shape:
        (T,H,W)

    We intentionally DO NOT resize here.
    esn.py performs nearest-neighbor resizing later.
    """
    if name not in group:
        raise KeyError(
            f"{group.name}: missing required dataset '{name}'. "
            "Use multimodal_object_context.h5 rather than the old "
            "RGB-only dataset."
        )

    mask = np.asarray(
        group[name][:],
        dtype=np.uint8,
    )

    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    if mask.ndim != 3:
        raise ValueError(
            f"{group.name}: {name} must have shape "
            f"(T,H,W) or (T,H,W,1); got {mask.shape}"
        )

    if len(mask) != length:
        raise ValueError(
            f"{group.name}: {name}/RGB frame mismatch: "
            f"{mask.shape} vs T={length}"
        )

    if tuple(mask.shape[1:]) != tuple(spatial_shape):
        raise ValueError(
            f"{group.name}: {name}/RGB spatial mismatch: "
            f"{mask.shape[1:]} vs {spatial_shape}"
        )

    # The dataset maker writes 0/1 uint8, but canonicalize defensively.
    mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    if int(mask.sum()) == 0:
        raise ValueError(
            f"{group.name}: {name} is completely empty"
        )

    return mask


# ============================================================
# TRAJECTORY LOADING
# ============================================================

def _read_group(
    group,
    key,
    split,
    include_rgb=True,
    include_masks=True,
):
    """
    Read one synchronized trajectory.

    Returned model-facing keys:
        id
        kind
        label
        split
        image
        joints
        goal
        cube_xyz
        agent_mask
        interaction_mask
        length
    """
    if "rgb" not in group:
        raise KeyError(
            f"{key}: missing rgb"
        )

    source_rgb = group["rgb"]

    if (
        source_rgb.ndim != 4
        or source_rgb.shape[-1] != 3
    ):
        raise ValueError(
            f"{key}: invalid RGB shape "
            f"{source_rgb.shape}"
        )

    length = len(source_rgb)
    native_h = int(
        source_rgb.shape[1]
    )
    native_w = int(
        source_rgb.shape[2]
    )

    joints = np.asarray(
        group["qpos"][:],
        dtype=np.float32,
    )

    goal = np.asarray(
        group["goal_xyz"][:],
        dtype=np.float32,
    ).reshape(3)

    cube = np.asarray(
        group["cube_xyz"][:],
        dtype=np.float32,
    )

    if joints.shape != (
        length,
        int(p.JOINT_DIM),
    ):
        raise ValueError(
            f"{key}: expected joints "
            f"({length},{p.JOINT_DIM}), "
            f"got {joints.shape}"
        )

    if cube.shape != (
        length,
        3,
    ):
        raise ValueError(
            f"{key}: RGB/cube mismatch: "
            f"{cube.shape}"
        )

    if not np.all(
        np.isfinite(joints)
    ):
        raise ValueError(
            f"{key}: qpos contains non-finite values"
        )

    if not np.all(
        np.isfinite(goal)
    ):
        raise ValueError(
            f"{key}: goal contains non-finite values"
        )

    if not np.all(
        np.isfinite(cube)
    ):
        raise ValueError(
            f"{key}: cube_xyz contains non-finite values"
        )

    label = int(
        group.attrs["label"]
    )

    if label == 0:
        kind = "pick"
    elif label == 1:
        kind = "push"
    else:
        raise ValueError(
            f"{key}: invalid label {label}"
        )

    images = (
        _process_rgb(
            source_rgb[:]
        )
        if include_rgb
        else None
    )

    if include_masks:
        agent_mask = _read_binary_mask(
            group,
            "agent_mask",
            length,
            (native_h, native_w),
        )

        interaction_mask = _read_binary_mask(
            group,
            "interaction_mask",
            length,
            (native_h, native_w),
        )
    else:
        agent_mask = None
        interaction_mask = None

    # Useful sanity check: generic masks should not overlap heavily.
    # A small overlap can occur around occlusion boundaries depending on
    # renderer semantics, so we do not reject overlap outright.
    if include_masks:
        overlap = np.logical_and(
            agent_mask,
            interaction_mask,
        ).sum()

        if overlap > 0:
            # Keep this informational only; visible segmentation is
            # normally exclusive per pixel.
            pass

    return {
        "id": key,
        "kind": kind,
        "label": label,
        "split": split,

        "image": images,
        "joints": joints,
        "goal": goal,
        "cube_xyz": cube,

        "agent_mask": agent_mask,
        "interaction_mask": interaction_mask,

        "length": length,
        "native_image_shape": (
            native_h,
            native_w,
        ),
    }


# ============================================================
# PUBLIC LOADERS
# ============================================================

def load_split(
    split,
    data_path=None,
    include_rgb=True,
    include_masks=True,
):
    """
    Load all episodes in one split.

    No temporal resampling.
    No validation/test information is used for normalization.
    """
    if split not in (
        "train",
        "validation",
        "test",
    ):
        raise ValueError(
            "split must be train, validation or test"
        )

    path = (
        Path(data_path).expanduser()
        if data_path is not None
        else DEFAULT_PATH.expanduser()
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    trajectories = []

    with h5py.File(
        path,
        "r",
    ) as handle:
        if "trajectories" not in handle:
            raise KeyError(
                "Expected /trajectories in "
                + str(path)
            )

        root = handle[
            "trajectories"
        ]

        for key in sorted(root):
            group = root[key]

            group_split = _attr_text(
                group.attrs["split"]
            )

            if group_split != split:
                continue

            trajectories.append(
                _read_group(
                    group,
                    key,
                    split,
                    include_rgb=include_rgb,
                    include_masks=include_masks,
                )
            )

    if not trajectories:
        raise ValueError(
            f"No trajectories for split "
            f"{split} in {path}"
        )

    return trajectories


def load_trajectory(
    data_path,
    trajectory_id,
    include_rgb=True,
    include_masks=True,
):
    """
    Retrieve one episode by ID.

    Example:
        pick_000
        push_025
    """
    path = Path(
        data_path
    ).expanduser()

    name = str(
        trajectory_id
    )

    with h5py.File(
        path,
        "r",
    ) as handle:
        group = handle[
            "trajectories"
        ][name]

        return _read_group(
            group,
            name,
            _attr_text(
                group.attrs["split"]
            ),
            include_rgb=include_rgb,
            include_masks=include_masks,
        )


# ============================================================
# TRAIN-ONLY NORMALIZATION
# ============================================================

def compute_train_stats(train):
    """
    Compute normalization statistics from TRAIN only.

    Masks are binary supervision targets and are intentionally
    not normalized.
    """
    joints = np.concatenate(
        [
            row["joints"]
            for row in train
        ],
        axis=0,
    )

    goals = np.stack(
        [
            row["goal"]
            for row in train
        ]
    )

    qmean = joints.mean(
        axis=0
    ).astype(
        np.float32
    )

    qstd = np.maximum(
        joints.std(axis=0),
        1e-6,
    ).astype(
        np.float32
    )

    gmean = goals.mean(
        axis=0
    ).astype(
        np.float32
    )

    gstd = np.maximum(
        goals.std(axis=0),
        1e-6,
    ).astype(
        np.float32
    )

    return {
        "qmean": qmean,
        "qstd": qstd,
        "gmean": gmean,
        "gstd": gstd,
    }


def normalize_modalities(
    data,
    stats,
):
    """
    Return copies with normalized joint/goal data.

    RGB and masks are unchanged.

    Original joint/goal arrays are retained in:
        raw_joints
        raw_goal
    """
    result = []

    for row in data:
        item = dict(row)

        item[
            "raw_joints"
        ] = row[
            "joints"
        ]

        item[
            "raw_goal"
        ] = row[
            "goal"
        ]

        item[
            "joints"
        ] = (
            (
                row["joints"]
                - stats["qmean"]
            )
            / stats["qstd"]
        ).astype(
            np.float32
        )

        item[
            "goal"
        ] = (
            (
                row["goal"]
                - stats["gmean"]
            )
            / stats["gstd"]
        ).astype(
            np.float32
        )

        # agent_mask and interaction_mask remain untouched.
        result.append(item)

    return result


# ============================================================
# SIMPLE DATASET SANITY REPORT
# ============================================================

def summarize_split(data):
    """
    Small helper useful before training.
    """
    lengths = np.asarray(
        [
            row["length"]
            for row in data
        ],
        dtype=np.int32,
    )

    agent_pixels = np.concatenate(
        [
            row["agent_mask"]
            .reshape(
                row["length"],
                -1,
            )
            .sum(axis=1)
            for row in data
        ]
    )

    interaction_pixels = np.concatenate(
        [
            row["interaction_mask"]
            .reshape(
                row["length"],
                -1,
            )
            .sum(axis=1)
            for row in data
        ]
    )

    return {
        "episodes": len(data),
        "min_frames": int(lengths.min()),
        "max_frames": int(lengths.max()),
        "mean_agent_pixels": float(
            agent_pixels.mean()
        ),
        "mean_interaction_pixels": float(
            interaction_pixels.mean()
        ),
        "empty_agent_frames": int(
            np.sum(agent_pixels == 0)
        ),
        "empty_interaction_frames": int(
            np.sum(
                interaction_pixels == 0
            )
        ),
    }