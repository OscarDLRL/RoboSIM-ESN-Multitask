from pathlib import Path

import numpy as np
import cupy as cp

from models.motion_only import parameters as p

# Reutilizamos solamente la selección de contexto y el reservoir
# ya validados en V14b.
from query_v14 import (
    context_indices,
    query_targets,
    _v13_context,
)


QUERY_VERSION = "v14c_residual_nearest_anchor"

META_DIM = 7

# base
# base*t
# base*t^2
# base*dt
# base*dt^2
# base*sin(pi*dt)
# base*cos(pi*dt)
N_BASE_COPIES = 7


# ============================================================
# ANCHOR
# ============================================================

def nearest_anchor(obs_idx, target):
    """
    Choose the temporally nearest available observation.
    """
    obs_idx = np.asarray(
        obs_idx,
        dtype=np.int32,
    )

    target = int(target)

    pos = int(
        np.argmin(
            np.abs(
                obs_idx - target
            )
        )
    )

    return int(
        obs_idx[pos]
    )


# ============================================================
# BASE FEATURE
# ============================================================

def residual_base_feature(
    row,
    model,
    context,
    obs_idx,
    target,
):
    """
    Base feature contains:

        global sparse V13 context
            reservoir
            last observed RGB PCA
            last observed q

        +

        nearest anchor RGB PCA
        nearest anchor q
    """

    obs_idx = np.asarray(
        obs_idx,
        dtype=np.int32,
    )

    anchor = nearest_anchor(
        obs_idx,
        target,
    )

    qmean = np.asarray(
        model["stats"]["qmean"],
        np.float32,
    )

    qstd = np.asarray(
        model["stats"]["qstd"],
        np.float32,
    )

    first_k = sorted(
        model["heads"]
    )[0]

    image_mean = model[
        "heads"
    ][first_k]["image_mean"]

    basis = model[
        "bases"
    ]["all"]

    anchor_image = cp.asarray(
        row["image"][anchor],
        cp.float32,
    )

    anchor_rgb_pca = (
        anchor_image
        - image_mean
    ) @ basis

    anchor_q = cp.asarray(
        (
            np.asarray(
                row["joints"][anchor],
                np.float32,
            )
            - qmean
        )
        / qstd,
        cp.float32,
    )

    base = cp.concatenate(
        [
            context,
            anchor_rgb_pca,
            anchor_q,
        ]
    )

    return base, anchor


# ============================================================
# QUERY FEATURE
# ============================================================

def residual_query_raw_from_context(
    row,
    model,
    context,
    obs_idx,
    target,
):
    """
    V14c feature.

    The important variable is now:

        dt = t_target - t_anchor

    rather than trying to generate the complete absolute state
    directly.
    """

    obs_idx = np.asarray(
        obs_idx,
        dtype=np.int32,
    )

    target = int(target)

    T = int(
        len(row["image"])
    )

    base, anchor = residual_base_feature(
        row,
        model,
        context,
        obs_idx,
        target,
    )

    denom = np.float32(
        max(T - 1, 1)
    )

    tau = np.float32(
        target / denom
    )

    tau2 = np.float32(
        tau * tau
    )

    anchor_t = np.float32(
        anchor / denom
    )

    delta = np.float32(
        (target - anchor)
        / denom
    )

    delta2 = np.float32(
        delta * delta
    )

    first = np.float32(
        obs_idx[0]
        / denom
    )

    last = np.float32(
        obs_idx[-1]
        / denom
    )

    span = np.float32(
        last - first
    )

    nobs = np.float32(
        len(obs_idx)
        / 5.0
    )

    sin_delta = np.float32(
        np.sin(
            np.pi * delta
        )
    )

    cos_delta = np.float32(
        np.cos(
            np.pi * delta
        )
    )

    parts = [
        base,

        base * tau,
        base * tau2,

        base * delta,
        base * delta2,

        base * sin_delta,
        base * cos_delta,
    ]

    meta = cp.asarray(
        [
            tau,
            anchor_t,
            delta,
            first,
            last,
            span,
            nobs,
        ],
        cp.float32,
    )

    parts.append(
        meta
    )

    return (
        cp.concatenate(parts),
        anchor,
    )


def residual_query_raw_feature(
    row,
    model,
    obs_idx,
    target,
):
    context = _v13_context(
        row,
        model,
        obs_idx,
    )

    return residual_query_raw_from_context(
        row,
        model,
        context,
        obs_idx,
        target,
    )


# ============================================================
# FEATURE BLOCKS
# ============================================================

def query_block_sizes(model):
    """
    Needed by the V13 block-normalization machinery.
    """

    N = int(
        model["N"]
    )

    rank = int(
        model[
            "bases"
        ]["all"].shape[1]
    )

    # V13 compact context:
    #
    # reservoir
    # + last RGB PCA
    # + last q
    #
    # anchor:
    # + anchor RGB PCA
    # + anchor q
    base_blocks = [
        ("state", N),

        ("small", rank),
        ("q", int(p.JOINT_DIM)),

        ("small", rank),
        ("q", int(p.JOINT_DIM)),
    ]

    sizes = []

    for _ in range(
        N_BASE_COPIES
    ):
        sizes.extend(
            base_blocks
        )

    sizes.append(
        (
            "time",
            META_DIM,
        )
    )

    return sizes


# ============================================================
# SAVE / LOAD
# ============================================================

def save_query_head(
    path,
    head,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,

        version=QUERY_VERSION,

        fmean=head["fmean"],
        fscale=head["fscale"],

        q_W=head["q_W"],
        q_b=head["q_b"],

        rgb_W=head["rgb_W"],
        rgb_b=head["rgb_b"],

        basis=head["basis"],

        qmean=head["qmean"],
        qstd=head["qstd"],
    )


def load_query_head(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as d:

        version = str(
            d["version"]
        )

        if version != QUERY_VERSION:
            raise ValueError(
                f"Wrong query head version: "
                f"{version}"
            )

        return {
            "fmean":
                cp.asarray(
                    d["fmean"],
                    cp.float32,
                ),

            "fscale":
                cp.asarray(
                    d["fscale"],
                    cp.float32,
                ),

            "q_W":
                cp.asarray(
                    d["q_W"],
                    cp.float32,
                ),

            "q_b":
                cp.asarray(
                    d["q_b"],
                    cp.float32,
                ),

            "rgb_W":
                cp.asarray(
                    d["rgb_W"],
                    cp.float32,
                ),

            "rgb_b":
                cp.asarray(
                    d["rgb_b"],
                    cp.float32,
                ),

            "basis":
                cp.asarray(
                    d["basis"],
                    cp.float32,
                ),

            "qmean":
                np.asarray(
                    d["qmean"],
                    np.float32,
                ),

            "qstd":
                np.asarray(
                    d["qstd"],
                    np.float32,
                ),
        }


# ============================================================
# PREDICTION
# ============================================================

def predict_query_from_context(
    row,
    model,
    head,
    context,
    obs_idx,
    target,
):
    obs_idx = np.asarray(
        obs_idx,
        dtype=np.int32,
    )

    target = int(target)

    # --------------------------------------------------------
    # If target is already observed:
    # don't predict what we already know.
    # --------------------------------------------------------
    if np.any(
        obs_idx == target
    ):
        return (
            np.asarray(
                row["image"][target],
                np.float32,
            ).copy(),

            np.asarray(
                row["joints"][target],
                np.float32,
            ).copy(),
        )

    raw, anchor = (
        residual_query_raw_from_context(
            row,
            model,
            context,
            obs_idx,
            target,
        )
    )

    x = (
        raw
        - head["fmean"]
    ) * head["fscale"]

    # --------------------------------------------------------
    # Predict DELTAS
    # --------------------------------------------------------
    dq_norm = (
        x @ head["q_W"]
        + head["q_b"]
    )

    drgb_coef = (
        x @ head["rgb_W"]
        + head["rgb_b"]
    )

    # --------------------------------------------------------
    # Anchor q + predicted delta
    # --------------------------------------------------------
    anchor_q = np.asarray(
        row["joints"][anchor],
        np.float32,
    )

    q_pred = (
        anchor_q
        + cp.asnumpy(
            dq_norm
        )
        * head["qstd"]
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Anchor image + predicted PCA delta
    # --------------------------------------------------------
    anchor_image = cp.asarray(
        row["image"][anchor],
        cp.float32,
    )

    image_pred = (
        anchor_image
        + drgb_coef
        @ head["basis"].T
    )

    image_pred = cp.clip(
        image_pred,
        0.0,
        1.0,
    )

    return (
        cp.asnumpy(
            image_pred
        ).astype(
            np.float32
        ),

        q_pred,
    )


def predict_query(
    row,
    model,
    head,
    obs_idx,
    target,
):
    context = _v13_context(
        row,
        model,
        obs_idx,
    )

    return predict_query_from_context(
        row,
        model,
        head,
        context,
        obs_idx,
        target,
    )
