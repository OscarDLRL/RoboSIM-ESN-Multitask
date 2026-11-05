"""Utilities for inference-time robustness tests.

Only the OBSERVED prefix is corrupted. Future targets remain clean.

RGB noise:
    Uniform noise in normalized image space [0, 1]:
        x_noisy = clip(x + U(-epsilon, +epsilon), 0, 1)

Joint noise:
    Gaussian noise scaled by the training-set joint standard deviation:
        q_noisy = q + Normal(0, sigma * qstd)

No flips, rotations, affine transforms, or training augmentation are used here.
"""

from __future__ import annotations

import zlib
import numpy as np


def deterministic_seed(base_seed: int, row_id: str, k: int, repeat: int = 0) -> int:
    """Stable seed for a trajectory/K/repeat combination."""
    token = f"{row_id}|K={int(k)}|R={int(repeat)}".encode("utf-8")
    offset = zlib.crc32(token) & 0xFFFFFFFF
    return (int(base_seed) + offset) & 0xFFFFFFFF


def add_rgb_uniform_noise(
    x: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add uniform RGB noise to normalized images in [0, 1]."""
    x = np.asarray(x, dtype=np.float32)

    epsilon = float(epsilon)
    if epsilon < 0:
        raise ValueError("RGB noise epsilon must be >= 0")

    if epsilon == 0:
        return x.copy()

    noise = rng.uniform(
        -epsilon,
        epsilon,
        size=x.shape,
    ).astype(np.float32)

    return np.clip(
        x + noise,
        0.0,
        1.0,
    ).astype(np.float32)


def add_joint_gaussian_noise(
    joints: np.ndarray,
    sigma: float,
    qstd: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian joint noise in units of the training q standard deviation."""
    joints = np.asarray(joints, dtype=np.float32)
    qstd = np.asarray(qstd, dtype=np.float32).reshape(-1)

    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("Joint noise sigma must be >= 0")

    if joints.ndim != 2:
        raise ValueError(f"Expected joints (T,D), got {joints.shape}")

    if joints.shape[1] != len(qstd):
        raise ValueError(
            f"Joint dimension mismatch: joints={joints.shape[1]}, qstd={len(qstd)}"
        )

    if sigma == 0:
        return joints.copy()

    scale = sigma * np.maximum(qstd, 1e-8)

    noise = rng.normal(
        loc=0.0,
        scale=scale[None, :],
        size=joints.shape,
    ).astype(np.float32)

    return (joints + noise).astype(np.float32)


def corrupt_observation(
    row: dict,
    k: int,
    qstd: np.ndarray,
    rgb_epsilon: float = 0.0,
    joint_sigma: float = 0.0,
    seed: int = 123,
    repeat: int = 0,
) -> dict:
    """Return a copy of ``row`` with noise only in the first K observed frames.

    The clean input row is never modified.
    """
    k = int(k)

    image = np.asarray(row["image"], dtype=np.float32)
    joints = np.asarray(row["joints"], dtype=np.float32)

    if not 1 <= k <= len(image):
        raise ValueError(f"K must be in 1..{len(image)}, got {k}")

    if len(image) != len(joints):
        raise ValueError(
            f"{row.get('id', '?')}: image/joints length mismatch "
            f"{len(image)} vs {len(joints)}"
        )

    out = dict(row)
    noisy_image = image.copy()
    noisy_joints = joints.copy()

    rng = np.random.default_rng(
        deterministic_seed(
            seed,
            str(row.get("id", "")),
            k,
            repeat,
        )
    )

    if float(rgb_epsilon) > 0:
        noisy_image[:k] = add_rgb_uniform_noise(
            image[:k],
            rgb_epsilon,
            rng,
        )

    if float(joint_sigma) > 0:
        noisy_joints[:k] = add_joint_gaussian_noise(
            joints[:k],
            joint_sigma,
            qstd,
            rng,
        )

    out["image"] = noisy_image
    out["joints"] = noisy_joints
    out["_robustness_rgb_epsilon"] = float(rgb_epsilon)
    out["_robustness_joint_sigma"] = float(joint_sigma)
    out["_robustness_repeat"] = int(repeat)

    return out


def robustness_tag(rgb_epsilon: float, joint_sigma: float) -> str:
    """Filesystem-safe name for one corruption condition."""
    rgb = f"{float(rgb_epsilon):.4f}".rstrip("0").rstrip(".") or "0"
    jnt = f"{float(joint_sigma):.4f}".rstrip("0").rstrip(".") or "0"

    return (
        f"rgb_{rgb.replace('.', 'p')}"
        f"_joint_{jnt.replace('.', 'p')}"
    )
