from pathlib import Path
import zlib
import numpy as np
import cupy as cp

from models.motion_only import parameters as p
from models.motion_only import esn


QUERY_VERSION = "v14b_bilinear_target_query"

QUERY_META_DIM = 6

# base context + six context/time interactions
N_CONTEXT_COPIES = 7


def context_indices(row_id, T, n_obs):
    n_obs = int(n_obs)

    if not 1 <= n_obs <= 5:
        raise ValueError("n_obs must be 1..5")

    seed = (
        zlib.crc32(
            f"{row_id}|T={T}|n={n_obs}|v14b".encode()
        )
        & 0xffffffff
    )

    rng = np.random.default_rng(seed)

    return np.sort(
        rng.choice(
            np.arange(T),
            size=n_obs,
            replace=False,
        )
    ).astype(np.int32)


def query_targets(row_id, obs_idx, T):
    """
    Return four query times:
      before context
      between observations
      after context
      arbitrary
    """
    obs_idx = np.asarray(
        obs_idx,
        np.int32,
    )

    seed = (
        zlib.crc32(
            f"{row_id}|targets|"
            f"{obs_idx.tolist()}|v14b".encode()
        )
        & 0xffffffff
    )

    rng = np.random.default_rng(seed)

    out = []

    # BEFORE
    first = int(obs_idx[0])

    if first > 0:
        out.append(
            int(rng.integers(0, first))
        )
    else:
        out.append(
            int(rng.integers(0, T))
        )

    # BETWEEN
    candidates = []

    for a, b in zip(
        obs_idx[:-1],
        obs_idx[1:],
    ):
        if int(b) - int(a) > 1:
            candidates.extend(
                range(
                    int(a) + 1,
                    int(b),
                )
            )

    if candidates:
        out.append(
            int(
                candidates[
                    rng.integers(
                        len(candidates)
                    )
                ]
            )
        )
    else:
        out.append(
            int(rng.integers(0, T))
        )

    # AFTER
    last = int(obs_idx[-1])

    if last < T - 1:
        out.append(
            int(
                rng.integers(
                    last + 1,
                    T,
                )
            )
        )
    else:
        out.append(
            int(rng.integers(0, T))
        )

    # RANDOM
    out.append(
        int(rng.integers(0, T))
    )

    return np.asarray(
        out,
        np.int32,
    )


def _v13_context(row, model, obs_idx):
    """
    Compact V13 context:

        reservoir state: N
        last RGB PCA:    48
        last q:           9

    For N=3000, PCA48:
        context_dim = 3057
    """
    obs_idx = np.unique(
        np.asarray(
            obs_idx,
            np.int32,
        )
    )

    if not len(obs_idx):
        raise ValueError(
            "At least one observation required"
        )

    T = len(row["image"])

    qmean = np.asarray(
        model["stats"]["qmean"],
        np.float32,
    )

    qstd = np.asarray(
        model["stats"]["qstd"],
        np.float32,
    )

    N = int(model["N"])

    state = cp.zeros(
        (N,),
        dtype=cp.float32,
    )

    WinT = cp.ascontiguousarray(
        model["Win"].T
    )

    WT = cp.ascontiguousarray(
        model["W"].T
    )

    bias = model["bias"]

    leak = float(p.LEAK_RATE)

    denom = np.float32(
        max(T - 1, 1)
    )

    # --------------------------------------------------------
    # Reservoir over only AVAILABLE observations
    # --------------------------------------------------------
    for prev, current in zip(
        obs_idx[:-1],
        obs_idx[1:],
    ):
        prev = int(prev)
        current = int(current)

        dimage = (
            float(
                model["vision_input_scale"]
            )
            * (
                np.asarray(
                    row["image"][current],
                    np.float32,
                )
                -
                np.asarray(
                    row["image"][prev],
                    np.float32,
                )
            )
        ).astype(np.float32)

        q0 = (
            np.asarray(
                row["joints"][prev],
                np.float32,
            )
            - qmean
        ) / qstd

        q1 = (
            np.asarray(
                row["joints"][current],
                np.float32,
            )
            - qmean
        ) / qstd

        dq = (
            float(p.JOINT_INPUT_SCALE)
            * (q1 - q0)
        ).astype(np.float32)

        dt = np.asarray(
            [
                (current - prev)
                / denom
            ],
            np.float32,
        )

        tnow = np.asarray(
            [
                current
                / denom
            ],
            np.float32,
        )

        u = cp.asarray(
            np.concatenate(
                [
                    dimage,
                    dq,
                    dt,
                    tnow,
                ]
            ),
            cp.float32,
        )

        candidate = cp.tanh(
            bias
            + u @ WinT
            + state @ WT
        )

        state = (
            (1.0 - leak) * state
            + leak * candidate
        )

    # --------------------------------------------------------
    # Absolute multimodal state at last observed instant
    # --------------------------------------------------------
    last = int(obs_idx[-1])

    image = cp.asarray(
        row["image"][last],
        cp.float32,
    )

    first_k = sorted(
        model["heads"]
    )[0]

    image_mean = model[
        "heads"
    ][first_k]["image_mean"]

    basis = model["bases"]["all"]

    rgb_pca = (
        image
        - image_mean
    ) @ basis

    q_last = cp.asarray(
        (
            np.asarray(
                row["joints"][last],
                np.float32,
            )
            - qmean
        ) / qstd,
        cp.float32,
    )

    return cp.concatenate(
        [
            state,
            rgb_pca,
            q_last,
        ]
    )


def query_raw_feature(
    row,
    model,
    obs_idx,
    target_index,
):
    """
    Bilinear time-conditioned query.

    This is the critical V14b change.

    A plain linear model:
        y = W_context*c + W_time*t

    cannot make the trajectory-specific effect of time depend on c.

    Here we give Ridge explicit products:
        c*t
        c*t²
        c*Δt
        c*Δt²
        c*sin(pi*Δt)
        c*cos(pi*Δt)
    """
    obs_idx = np.asarray(
        obs_idx,
        np.int32,
    )

    T = int(len(row["image"]))

    target_index = int(
        target_index
    )

    context = _v13_context(
        row,
        model,
        obs_idx,
    )

    denom = np.float32(
        max(T - 1, 1)
    )

    tau = np.float32(
        target_index / denom
    )

    first = np.float32(
        obs_idx[0] / denom
    )

    last = np.float32(
        obs_idx[-1] / denom
    )

    span = np.float32(
        last - first
    )

    delta = np.float32(
        tau - last
    )

    nobs = np.float32(
        len(obs_idx) / 5.0
    )

    gates = (
        tau,
        tau * tau,
        delta,
        delta * delta,
        np.float32(
            np.sin(
                np.pi * delta
            )
        ),
        np.float32(
            np.cos(
                np.pi * delta
            )
        ),
    )

    parts = [
        context
    ]

    for g in gates:
        parts.append(
            context
            * cp.float32(g)
        )

    meta = cp.asarray(
        [
            tau,
            first,
            last,
            span,
            delta,
            nobs,
        ],
        cp.float32,
    )

    parts.append(meta)

    return cp.concatenate(
        parts
    )


def query_block_sizes(model):
    """
    Block structure for feature normalization.
    """
    N = int(model["N"])

    rank = int(
        model["bases"]["all"].shape[1]
    )

    base = [
        ("state", N),
        ("small", rank),
        ("q", p.JOINT_DIM),
    ]

    out = []

    for _ in range(
        N_CONTEXT_COPIES
    ):
        out.extend(base)

    out.append(
        ("time", QUERY_META_DIM)
    )

    return out


def save_query_head(path, head):
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
        image_mean=head["image_mean"],

        qmean=head["qmean"],
        qstd=head["qstd"],
    )


def load_query_head(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as d:

        if str(
            d["version"]
        ) != QUERY_VERSION:
            raise ValueError(
                "Wrong V14 query version"
            )

        return {
            key: cp.asarray(
                d[key],
                cp.float32,
            )
            for key in (
                "fmean",
                "fscale",
                "q_W",
                "q_b",
                "rgb_W",
                "rgb_b",
                "basis",
                "image_mean",
            )
        } | {
            "qmean": np.asarray(
                d["qmean"],
                np.float32,
            ),

            "qstd": np.asarray(
                d["qstd"],
                np.float32,
            ),
        }


def predict_query(
    row,
    model,
    head,
    obs_idx,
    target_index,
):
    raw = query_raw_feature(
        row,
        model,
        obs_idx,
        target_index,
    )

    x = (
        raw
        - head["fmean"]
    ) * head["fscale"]

    qn = (
        x @ head["q_W"]
        + head["q_b"]
    )

    coeff = (
        x @ head["rgb_W"]
        + head["rgb_b"]
    )

    q = (
        head["qmean"]
        + cp.asnumpy(qn)
        * head["qstd"]
    ).astype(np.float32)

    image = (
        head["image_mean"]
        + coeff
        @ head["basis"].T
    )

    image = cp.clip(
        image,
        0.0,
        1.0,
    )

    return (
        cp.asnumpy(image).astype(
            np.float32
        ),
        q,
    )
