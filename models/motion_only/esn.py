from pathlib import Path
import zlib
import numpy as np
import cupy as cp
from . import parameters as p

ARCHITECTURE = "maniskill_hybrid_context_v13_deltaq_matrixfree"
TRAINING_PIPELINE = "hybrid_history_sparse_1to5_randomtime_v5_matrixfree_ridge"
HEAD_ARRAYS = ("fmean", "fscale", "rgb_W", "rgb_b", "image_mean", "anchor_W", "anchor_b")


def input_width():
    if p.EXPERIMENT_MODE != "dropout":
        raise ValueError("Final motion_only model requires EXPERIMENT_MODE='dropout'")
    return p.IMAGE_DIM + p.JOINT_DIM + 2


def _block_weights():
    d = {"state": 1.0, "small": 1.0, "q": 2.0}
    d.update(getattr(p, "BLOCK_WEIGHTS", {}))
    return {k: float(d[k]) for k in ("state", "small", "q")}


def _ridge_grid():
    return np.asarray(getattr(p, "RIDGE_GRID", np.logspace(-10, 1, 12)), np.float64)


def _dropout_mask(mode):
    mode = str(mode)
    if mode == "full":
        return 1.0, 1.0
    if mode == "vision":
        return 1.0, 0.0
    if mode == "joints":
        return 0.0, 1.0
    raise ValueError(f"Unknown modality condition: {mode}")


def _row_mask(row, model):
    if "_modality_mask" in row:
        m = np.asarray(row["_modality_mask"], np.float32).reshape(2)
        return float(m[0]), float(m[1])
    return _dropout_mask(getattr(p, "MODALITY_TEST_MODE", "full"))


def _dropout_pattern():
    pattern = getattr(p, "MODALITY_DROPOUT_PATTERN", (("full", 2), ("vision", 1), ("joints", 1)))
    out = []
    for name, count in pattern:
        _dropout_mask(name)
        count = int(count)
        if count < 0:
            raise ValueError("Dropout copy counts must be >= 0")
        if count:
            out.append((str(name), count))
    if not out:
        raise ValueError("MODALITY_DROPOUT_PATTERN cannot be empty")
    return tuple(out)


def _expand_dropout_rows(rows):
    pattern = _dropout_pattern(); out = []
    for row in rows:
        source_id = str(row["id"])
        for condition, count in pattern:
            mask = np.asarray(_dropout_mask(condition), np.float32)
            for copy_index in range(count):
                clone = dict(row)
                clone["_modality_mask"] = mask
                clone["_modality_condition"] = condition
                clone["_source_id"] = source_id
                clone["_dropout_copy"] = int(copy_index)
                clone["_hybrid_context_train"] = bool(
                    getattr(p, "HYBRID_CONTEXT_TRAIN", False)
                )
                out.append(clone)
    return out


def _group_labels(rows):
    return np.asarray([str(row.get("_source_id", row["id"])) for row in rows])


def _obs_image(row):
    return row.get("_obs_image", row["image"])


def _obs_joints(row):
    return row.get("_obs_joints", row["joints"])


def _stable_noise_seed(base_seed, *parts):
    token = "|".join(str(x) for x in parts).encode("utf-8")
    return (int(base_seed) + (zlib.crc32(token) & 0xFFFFFFFF)) & 0xFFFFFFFF


def _add_rgb_uniform_noise(image, epsilon, rng):
    image = np.asarray(image, np.float32)
    epsilon = float(epsilon)
    if epsilon <= 0:
        return image.copy()
    noise = rng.uniform(-epsilon, epsilon, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


def _add_joint_gaussian_noise(joints, sigma, qstd, rng):
    joints = np.asarray(joints, np.float32)
    qstd = np.asarray(qstd, np.float32).reshape(-1)
    sigma = float(sigma)
    if sigma <= 0:
        return joints.copy()
    scale = sigma * np.maximum(qstd, 1e-8)
    noise = rng.normal(loc=0.0, scale=scale[None, :], size=joints.shape).astype(np.float32)
    return (joints + noise).astype(np.float32)


def _training_noise_levels():
    levels = tuple(float(x) for x in getattr(p, "TRAIN_NOISE_LEVELS", (0.01, 0.025, 0.05)))
    if not levels:
        raise ValueError("TRAIN_NOISE_LEVELS cannot be empty when TRAIN_SENSOR_NOISE=True")
    if any(x <= 0 for x in levels):
        raise ValueError("TRAIN_NOISE_LEVELS must contain only values > 0")
    return levels


def _training_noise_conditions():
    allowed = {"rgb", "joints", "both"}
    conditions = tuple(str(x) for x in getattr(p, "TRAIN_NOISE_CONDITIONS", ("rgb", "joints", "both")))
    if not conditions:
        raise ValueError("TRAIN_NOISE_CONDITIONS cannot be empty")
    bad = [x for x in conditions if x not in allowed]
    if bad:
        raise ValueError(f"Unknown TRAIN_NOISE_CONDITIONS: {bad}")
    return conditions


def _choose_training_noise_condition(row, copy_index):
    modality = str(row.get("_modality_condition", "full"))
    if modality == "vision":
        return "rgb"
    if modality == "joints":
        return "joints"
    conditions = _training_noise_conditions(); source_id = str(row.get("_source_id", row.get("id", ""))); dropout_copy = int(row.get("_dropout_copy", 0))
    h = zlib.crc32(f"{source_id}|drop={dropout_copy}|aug={int(copy_index)}".encode("utf-8")) & 0xFFFFFFFF
    return conditions[h % len(conditions)]


def _choose_training_noise_level(row, copy_index):
    levels = _training_noise_levels(); source_id = str(row.get("_source_id", row.get("id", ""))); modality = str(row.get("_modality_condition", "full")); dropout_copy = int(row.get("_dropout_copy", 0))
    h = zlib.crc32(f"{source_id}|{modality}|drop={dropout_copy}|aug={int(copy_index)}|level".encode("utf-8")) & 0xFFFFFFFF
    return levels[h % len(levels)]


def _make_noisy_training_row(row, stats, copy_index, seed):
    condition = _choose_training_noise_condition(row, copy_index)
    level = _choose_training_noise_level(row, copy_index)
    source_id = str(row.get("_source_id", row.get("id", "")))
    modality = str(row.get("_modality_condition", "full"))
    dropout_copy = int(row.get("_dropout_copy", 0))
    rng = np.random.default_rng(_stable_noise_seed(seed, source_id, modality, dropout_copy, copy_index, condition, level))
    clean_image = np.asarray(row["image"], np.float32)
    clean_joints = np.asarray(row["joints"], np.float32)
    obs_image = clean_image.copy(); obs_joints = clean_joints.copy()
    if condition in ("rgb", "both"):
        obs_image = _add_rgb_uniform_noise(obs_image, level, rng)
    if condition in ("joints", "both"):
        obs_joints = _add_joint_gaussian_noise(obs_joints, level, stats["qstd"], rng)
    clone = dict(row)
    clone["_obs_image"] = obs_image; clone["_obs_joints"] = obs_joints
    clone["_source_id"] = str(row.get("_source_id", row["id"]))
    clone["_training_noise"] = condition; clone["_training_noise_level"] = float(level); clone["_training_noise_copy"] = int(copy_index)
    return clone


def _training_specs(dropout_rows):
    if not bool(getattr(p, "TRAIN_SENSOR_NOISE", False)):
        return [(row, -1) for row in dropout_rows]
    copies = int(getattr(p, "TRAIN_NOISE_COPIES", 1))
    if copies < 1:
        raise ValueError("TRAIN_NOISE_COPIES must be >= 1 when TRAIN_SENSOR_NOISE=True")
    specs = []
    for row in dropout_rows:
        specs.append((row, -1))
        for c in range(copies):
            specs.append((row, c))
    return specs


def _materialize_spec(spec, stats, seed):
    row, copy_index = spec
    if copy_index < 0:
        return row
    return _make_noisy_training_row(row, stats, copy_index, seed)


def _validate(row):
    image = np.asarray(row["image"])
    if image.ndim != 2 or image.shape[1] != p.IMAGE_DIM:
        raise ValueError(f"{row['id']}: expected image (T,{p.IMAGE_DIM}), got {image.shape}")
    if len(image) != len(row["joints"]):
        raise ValueError(f"{row['id']}: image/joint frame mismatch")
    return image


def new_model(N, seed, stats):
    rng = cp.random.RandomState(int(seed)); dim = input_width(); Win = rng.uniform(-p.INPUT_SCALE, p.INPUT_SCALE, (N, dim)).astype(cp.float32)
    bias = rng.uniform(-p.INPUT_SCALE, p.INPUT_SCALE, (N,)).astype(cp.float32); W = rng.uniform(-1, 1, (N, N)).astype(cp.float32)
    radius = float(cp.max(cp.abs(cp.linalg.eigvals(W))).get()); W *= cp.float32(p.SPECTRAL_RADIUS / max(radius, 1e-12))
    return dict(Win=Win, W=W, bias=bias, N=N, seed=int(seed), mode="dropout", vision_input_scale=float(p.VISION_INPUT_SCALE), image_dim=p.IMAGE_DIM, stats=stats, heads={}, accuracy={}, bases={}, image_means={})




def _context_indices(row, k):
    """
    Temporal context visible to the ESN.

    Normal inference:
        complete history [0 ... K-1]

    Explicit sparse inference:
        row["_obs_indices"]

    V13 training:
        deterministic mixture of
        history / 1 / 2 / 3 / 5 observations.

    Sparse contexts always contain K-1, because the current
    model predicts the continuation K -> final.
    """
    k = int(k)

    explicit = row.get(
        "_obs_indices",
        None,
    )

    # --------------------------------------------------------
    # Explicit sparse test/inference
    # --------------------------------------------------------
    if explicit is not None:

        idx = np.asarray(
            explicit,
            np.int32,
        ).reshape(-1)

        idx = np.unique(idx)

        idx = idx[
            (idx >= 0)
            & (idx < k)
        ]

        if len(idx) == 0 or idx[-1] != k - 1:
            idx = np.concatenate([
                idx,
                np.asarray(
                    [k - 1],
                    np.int32,
                ),
            ])

        max_obs = int(
            getattr(
                p,
                "SPARSE_MAX_OBS",
                5,
            )
        )

        if len(idx) > max_obs:
            raise ValueError(
                f"At most {max_obs} sparse observations supported"
            )

        return idx.astype(np.int32), "explicit_sparse"

    # --------------------------------------------------------
    # Normal inference = full history
    # --------------------------------------------------------
    hybrid_train = bool(
        row.get(
            "_hybrid_context_train",
            False,
        )
    )

    if not hybrid_train:
        return (
            np.arange(
                k,
                dtype=np.int32,
            ),
            "history",
        )

    # --------------------------------------------------------
    # V13 deterministic hybrid assignment
    # --------------------------------------------------------
    source = str(
        row.get(
            "_source_id",
            row.get("id", ""),
        )
    )

    modality = str(
        row.get(
            "_modality_condition",
            "full",
        )
    )

    drop = int(
        row.get(
            "_dropout_copy",
            0,
        )
    )

    noise = int(
        row.get(
            "_training_noise_copy",
            -1,
        )
    )

    key = (
        f"{source}|{modality}|"
        f"drop={drop}|noise={noise}|K={k}|v13"
    )

    h = (
        zlib.crc32(
            key.encode("utf-8")
        )
        & 0xffffffff
    )

    pattern = tuple(
        getattr(
            p,
            "HYBRID_CONTEXT_PATTERN",
            ("history", 1, 2, 3, 5),
        )
    )

    choice = pattern[
        h % len(pattern)
    ]

    # Full dense history.
    if str(choice) == "history":
        return (
            np.arange(
                k,
                dtype=np.int32,
            ),
            "history",
        )

    n_obs = min(
        int(choice),
        k,
    )

    # Single current observation.
    if n_obs <= 1:
        return (
            np.asarray(
                [k - 1],
                np.int32,
            ),
            "sparse",
        )

    seed = (
        zlib.crc32(
            (key + "|indices").encode("utf-8")
        )
        & 0xffffffff
    )

    rng = np.random.default_rng(seed)

    previous = np.sort(
        rng.choice(
            np.arange(k - 1),
            size=n_obs - 1,
            replace=False,
        )
    ).astype(np.int32)

    idx = np.concatenate([
        previous,
        np.asarray(
            [k - 1],
            np.int32,
        ),
    ])

    return idx, "sparse"


def prepare_observation(row, model, k):

    vision_on, joints_on = _row_mask(
        row,
        model,
    )

    idx, context_mode = _context_indices(
        row,
        k,
    )

    ntrans = max(
        0,
        len(idx) - 1,
    )

    # --------------------------------------------------------
    # RGB differences between AVAILABLE observations
    # --------------------------------------------------------
    if vision_on == 0.0:

        dimage = cp.zeros(
            (ntrans, p.IMAGE_DIM),
            dtype=cp.float32,
        )

    else:

        image = cp.asarray(
            _obs_image(row)[idx],
            cp.float32,
        )

        dimage = (
            cp.float32(
                model["vision_input_scale"]
            )
            * (
                image[1:]
                - image[:-1]
            )
        )

    # --------------------------------------------------------
    # Joint differences between AVAILABLE observations
    # --------------------------------------------------------
    if joints_on == 0.0:

        dq = cp.zeros(
            (ntrans, p.JOINT_DIM),
            dtype=cp.float32,
        )

    else:

        raw_q = cp.asarray(
            _obs_joints(row)[idx],
            cp.float32,
        )

        q = (
            raw_q
            - cp.asarray(
                model["stats"]["qmean"]
            )
        ) / cp.asarray(
            model["stats"]["qstd"]
        )

        dq = (
            cp.float32(
                p.JOINT_INPUT_SCALE
            )
            * (
                q[1:]
                - q[:-1]
            )
        )

    # --------------------------------------------------------
    # Explicit timing
    # --------------------------------------------------------
    total_frames = max(
        2,
        len(_obs_image(row)),
    )

    denom = np.float32(
        total_frames - 1
    )

    if ntrans:

        dt = cp.asarray(
            np.diff(idx).astype(
                np.float32
            ) / denom,
            cp.float32,
        )[:, None]

        tnow = cp.asarray(
            idx[1:].astype(
                np.float32
            ) / denom,
            cp.float32,
        )[:, None]

    else:

        dt = cp.zeros(
            (0, 1),
            dtype=cp.float32,
        )

        tnow = cp.zeros(
            (0, 1),
            dtype=cp.float32,
        )

    return cp.concatenate(
        [
            dimage,
            dq,
            dt,
            tnow,
        ],
        axis=1,
    )


def block_sizes(model):
    return [("state", int(model["N"])), ("small", p.APPEARANCE_HEIGHT * p.APPEARANCE_WIDTH * p.IMAGE_CHANNELS), ("q", int(p.JOINT_DIM))]


def _tail_feature(row, model, k):
    vision_on, joints_on = _row_mask(row, model); c = p.IMAGE_CHANNELS; h, w = p.IMAGE_HEIGHT, p.IMAGE_WIDTH; ah, aw = p.APPEARANCE_HEIGHT, p.APPEARANCE_WIDTH
    if vision_on == 0.0:
        small = cp.zeros((ah * aw * c,), dtype=cp.float32)
    else:
        img = cp.asarray(_obs_image(row)[k - 1], cp.float32).reshape(h, w, c)
        small = img.reshape(ah, h // ah, aw, w // aw, c).mean(axis=(1, 3)).ravel() - cp.float32(0.5)
    if joints_on == 0.0:
        q = cp.zeros((p.JOINT_DIM,), dtype=cp.float32)
    else:
        q = (cp.asarray(_obs_joints(row)[k - 1], cp.float32) - cp.asarray(model["stats"]["qmean"])) / cp.asarray(model["stats"]["qstd"])
    return cp.concatenate([small, q])




def raw_features_batch(rows, model, k, batch=32):
    """
    Compute features efficiently for mixed temporal contexts.

    Rows are grouped by the ACTUAL number of reservoir updates:
        OBS1    -> 0 updates
        OBS2    -> 1
        OBS3    -> 2
        OBS5    -> 4
        HISTORY -> K-1

    Therefore sparse examples never execute fake padded timesteps.
    """
    N = int(model["N"])

    WinT = cp.ascontiguousarray(
        model["Win"].T
    )

    WT = cp.ascontiguousarray(
        model["W"].T
    )

    bias = model["bias"]
    leak = p.LEAK_RATE

    # Determine actual sequence length for every row.
    prepared = []

    for i, row in enumerate(rows):
        obs = prepare_observation(
            row,
            model,
            k,
        )

        prepared.append(
            (i, obs)
        )

    # Group by number of actual ESN transitions.
    groups = {}

    for i, obs in prepared:
        steps = int(
            obs.shape[0]
        )

        groups.setdefault(
            steps,
            [],
        ).append(
            (i, obs)
        )

    result = [None] * len(rows)

    for steps in sorted(groups):

        group = groups[steps]

        for start in range(
            0,
            len(group),
            int(batch),
        ):

            chunk = group[
                start:start + int(batch)
            ]

            indices = [
                item[0]
                for item in chunk
            ]

            B = len(chunk)

            state = cp.zeros(
                (B, N),
                dtype=cp.float32,
            )

            if steps:

                obs = cp.stack([
                    item[1]
                    for item in chunk
                ])

                for t in range(steps):

                    cand = cp.tanh(
                        bias[None, :]
                        + obs[:, t, :] @ WinT
                        + state @ WT
                    )

                    state = (
                        (1 - leak) * state
                        + leak * cand
                    )

            tails = cp.stack([
                _tail_feature(
                    rows[i],
                    model,
                    k,
                )
                for i in indices
            ])

            feat = cp.concatenate(
                [
                    state,
                    tails,
                ],
                axis=1,
            )

            for j, original_index in enumerate(indices):
                result[original_index] = feat[j]

    return cp.stack(result)


def raw_feature(row, model, k):
    return raw_features_batch([row], model, k, batch=1)[0]


def _training_features_for_k(specs, model, stats, seed, k, chunk=None):
    chunk = int(chunk if chunk is not None else getattr(p, "TRAIN_FEATURE_CHUNK", 32)); k = int(k); feats = []; idxs = []
    for s in range(0, len(specs), chunk):
        rows = [_materialize_spec(sp, stats, seed) for sp in specs[s:s + chunk]]
        ii = [i for i, r in enumerate(rows) if len(r["image"]) > k]
        if ii:
            batch_rows = [rows[i] for i in ii]
            feats.append(raw_features_batch(batch_rows, model, k, batch=len(batch_rows)))
            idxs.extend(s + i for i in ii)
        del rows
    if not feats:
        return np.asarray([], np.int64), None
    return np.asarray(idxs, np.int64), cp.concatenate(feats, axis=0)


def _norm(raw, head):
    return (raw - head["fmean"]) * head["fscale"]


def _image_basis(rows, rank, seed, absolute=False):
    rng = np.random.RandomState(int(seed)); imgs = [np.asarray(row["image"], np.float32) for row in rows]; pairs = []
    for i, im in enumerate(imgs):
        T = len(im)
        for k in p.READOUT_K:
            if T > k:
                pairs.extend((i, k, q) for q in range(k, T))
    max_samples = int(getattr(p, "IMAGE_BASIS_SAMPLES", 4000))
    if len(pairs) > max_samples:
        sel = rng.choice(len(pairs), max_samples, replace=False)
        pairs = [pairs[j] for j in sel]
    if absolute:
        D = cp.asarray(np.stack([imgs[i][q] for i, k, q in pairs]).astype(np.float32)); image_mean = D.mean(axis=0); D = D - image_mean
    else:
        D = cp.asarray(np.stack([imgs[i][q] - imgs[i][k - 1] for i, k, q in pairs]).astype(np.float32)); image_mean = cp.zeros((p.IMAGE_DIM,), dtype=cp.float32)
    _, _, Vt = cp.linalg.svd(D, full_matrices=False)
    rank = int(min(rank, Vt.shape[0]))
    basis = cp.ascontiguousarray(Vt[:rank].T).astype(cp.float32)
    return basis, image_mean.astype(cp.float32)


def _make_group_folds(groups, n_folds, seed):
    groups = np.asarray(groups); unique = np.unique(groups); n_folds = int(min(max(2, n_folds), len(unique))); rng = np.random.RandomState(int(seed)); unique = unique[rng.permutation(len(unique))]; buckets = np.array_split(unique, n_folds); folds = []
    for bucket in buckets:
        va = np.flatnonzero(np.isin(groups, bucket)); tr = np.flatnonzero(~np.isin(groups, bucket))
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


def _feature_normalizer(raw, sizes, weights, indices=None):
    R = raw if indices is None else raw[cp.asarray(indices, dtype=cp.int32)]
    fmean = R.mean(axis=0, dtype=cp.float64).astype(cp.float32); scales = []; start = 0
    for name, size in sizes:
        blk = R[:, start:start + size].astype(cp.float64) - fmean[start:start + size].astype(cp.float64)
        msq = float(cp.mean(cp.sum(blk * blk, axis=1)))
        sc = np.sqrt(float(weights[name]) / max(msq, 1e-12)); scales.append(cp.full((size,), sc, dtype=cp.float32)); start += size
    return fmean, cp.concatenate(scales)


def _apply_normalizer(raw, fmean, fscale, indices=None):
    R = raw if indices is None else raw[cp.asarray(indices, dtype=cp.int32)]
    return (R - fmean) * fscale


def _select_output_columns(q, max_cols, seed):
    q = int(q); max_cols = int(max_cols)
    if q <= max_cols:
        return np.arange(q, dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    return np.sort(rng.choice(q, max_cols, replace=False)).astype(np.int64)


def _cv_group_subset(groups, max_rows, seed):
    groups = np.asarray(groups); unique = np.unique(groups); rng = np.random.RandomState(int(seed)); unique = unique[rng.permutation(len(unique))]; picked = []; total = 0
    for g in unique:
        ix = np.flatnonzero(groups == g)
        if picked and total + len(ix) > int(max_rows):
            break
        picked.append(ix); total += len(ix)
    if not picked:
        picked = [np.flatnonzero(groups == unique[0])]
    return np.concatenate(picked).astype(np.int64)


def _select_ridge_relative_lambda(raw, Y, groups, sizes, weights, lam_grid, seed):
    groups = np.asarray(groups); max_rows = int(getattr(p, "RIDGE_CV_MAX_ROWS", 1200)); max_outputs = int(getattr(p, "RIDGE_CV_OUTPUTS", 64)); n_folds = int(getattr(p, "RIDGE_GROUP_FOLDS", 5))
    sub = _cv_group_subset(groups, max_rows, seed); sub_groups = groups[sub]; folds_local = _make_group_folds(sub_groups, n_folds, seed)
    Y = cp.asarray(Y, cp.float32); cols = _select_output_columns(Y.shape[1], max_outputs, seed + 37); Ysub = Y[cp.asarray(sub, dtype=cp.int32)][:, cp.asarray(cols, dtype=cp.int32)]
    rel_grid = np.asarray(lam_grid, np.float64); sse = np.zeros(len(rel_grid), np.float64); count = np.zeros(len(rel_grid), np.int64); raw_sub = raw[cp.asarray(sub, dtype=cp.int32)]
    for fold_id, (tr_local, va_local) in enumerate(folds_local):
        fmean, fscale = _feature_normalizer(raw_sub, sizes, weights, tr_local)
        Xtr = _apply_normalizer(raw_sub, fmean, fscale, tr_local).astype(cp.float32)
        Xva = _apply_normalizer(raw_sub, fmean, fscale, va_local).astype(cp.float32)
        tr_cp = cp.asarray(tr_local, dtype=cp.int32); va_cp = cp.asarray(va_local, dtype=cp.int32); Ytr = Ysub[tr_cp]; Yva = Ysub[va_cp]
        ymean = Ytr.mean(axis=0); Ytrc = Ytr - ymean; Yvac = Yva - ymean
        Ktr = (Xtr @ Xtr.T).astype(cp.float64); Kva = (Xva @ Xtr.T).astype(cp.float64); s, V = cp.linalg.eigh(Ktr); s = cp.maximum(s, 0.0); VtY = V.T @ Ytrc.astype(cp.float64)
        scale0 = max(float(cp.mean(cp.sum(Xtr.astype(cp.float64) ** 2, axis=1))), 1e-12)
        for i, rel in enumerate(rel_grid):
            lam = scale0 * float(rel); alpha = V @ (VtY / (s + lam)[:, None]); pred = Kva @ alpha; err = pred - Yvac.astype(cp.float64); sse[i] += float(cp.sum(err * err)); count[i] += int(err.size)
        del Xtr, Xva, Ytr, Yva, Ytrc, Yvac, Ktr, Kva, s, V, VtY; cp.get_default_memory_pool().free_all_blocks()
    mse = sse / np.maximum(count, 1); best = int(np.argmin(mse)); return float(rel_grid[best]), float(np.sqrt(mse[best]))


def _pcg(A, B, M_inv, tol, max_iter):
    B = cp.asarray(B, cp.float32); X = cp.zeros_like(B); R = B.copy(); Z = M_inv[:, None] * R; P = Z.copy(); rz = cp.sum(R * Z, axis=0); bnorm = cp.sqrt(cp.sum(B * B, axis=0)); bnorm = cp.maximum(bnorm, cp.float32(1e-12)); active = cp.ones((B.shape[1],), dtype=cp.bool_); it_used = 0
    for it in range(int(max_iter)):
        AP = A(P); denom = cp.sum(P * AP, axis=0); denom = cp.where(cp.abs(denom) > 1e-20, denom, cp.float32(1e-20)); alpha = cp.where(active, rz / denom, cp.float32(0.0)); X += P * alpha[None, :]; R -= AP * alpha[None, :]
        rel = cp.sqrt(cp.sum(R * R, axis=0)) / bnorm; active_new = rel > float(tol); it_used = it + 1
        if not bool(cp.any(active_new)):
            active = active_new; break
        Z = M_inv[:, None] * R; rz_new = cp.sum(R * Z, axis=0); beta = cp.where(active_new & (cp.abs(rz) > 1e-20), rz_new / rz, cp.float32(0.0)); P = Z + P * beta[None, :]; active = active_new; rz = rz_new
    final_rel = float(cp.max(cp.sqrt(cp.sum(R * R, axis=0)) / bnorm))
    return X, it_used, final_rel


def _ridge_matrixfree_fit(X, Y, ridge_rel):
    X = cp.asarray(X, cp.float32); Y = cp.asarray(Y, cp.float32); n, D = X.shape; ymean = Y.mean(axis=0); Yc = Y - ymean; scale0 = max(float(cp.mean(cp.sum(X.astype(cp.float64) ** 2, axis=1))), 1e-12); lam = float(scale0 * float(ridge_rel)); tol = float(getattr(p, "RIDGE_CG_TOL", 1e-4)); max_iter = int(getattr(p, "RIDGE_CG_MAX_ITER", 80)); out_chunk = int(getattr(p, "RIDGE_OUTPUT_CHUNK", 128)); W = cp.empty((D, Y.shape[1]), dtype=cp.float32); max_used = 0; max_resid = 0.0
    if n <= D:
        diag = cp.sum(X * X, axis=1) + cp.float32(lam); M_inv = 1.0 / cp.maximum(diag, cp.float32(1e-8))
        def A(P):
            return X @ (X.T @ P) + cp.float32(lam) * P
        for s in range(0, Y.shape[1], out_chunk):
            B = Yc[:, s:s + out_chunk]; alpha, used, resid = _pcg(A, B, M_inv, tol, max_iter); W[:, s:s + B.shape[1]] = X.T @ alpha; max_used = max(max_used, used); max_resid = max(max_resid, resid); del B, alpha
        solver = "dual-pcg"
    else:
        diag = cp.sum(X * X, axis=0) + cp.float32(lam); M_inv = 1.0 / cp.maximum(diag, cp.float32(1e-8))
        def A(P):
            return X.T @ (X @ P) + cp.float32(lam) * P
        for s in range(0, Y.shape[1], out_chunk):
            B = X.T @ Yc[:, s:s + out_chunk]; Wc, used, resid = _pcg(A, B, M_inv, tol, max_iter); W[:, s:s + Wc.shape[1]] = Wc; max_used = max(max_used, used); max_resid = max(max_resid, resid); del B, Wc
        solver = "primal-pcg"
    return W, ymean, lam, solver, max_used, max_resid


def _fit_ridge_matrixfree(raw, Y, groups, sizes, weights, lam_grid, seed):
    rel, cv_rmse = _select_ridge_relative_lambda(raw, Y, groups, sizes, weights, lam_grid, seed)
    fmean, fscale = _feature_normalizer(raw, sizes, weights, indices=None); X = _apply_normalizer(raw, fmean, fscale).astype(cp.float32)
    W, b, lam, solver, iters, resid = _ridge_matrixfree_fit(X, Y, rel)
    return dict(fmean=fmean.astype(cp.float32), fscale=fscale.astype(cp.float32), X=X, W=W.astype(cp.float32), b=b.astype(cp.float32), ridge_rel=float(rel), lam=float(lam), cv_rmse=float(cv_rmse), solver=solver, cg_iters=int(iters), cg_resid=float(resid))


def _smooth(a, window):
    window = int(window)
    if window <= 1 or len(a) == 0:
        return a
    pad = window // 2; ap = cp.pad(a, ((pad, pad), (0, 0)), mode="edge")
    return sum(ap[i:i + len(a)] for i in range(2 * pad + 1)) / (2 * pad + 1)


def _fit_head(rows, raw, model, k, basis, image_mean, max_len, with_cls=False):
    qmean = np.asarray(model["stats"]["qmean"], np.float32); qstd = np.asarray(model["stats"]["qstd"], np.float32); weights, grid, sizes = _block_weights(), _ridge_grid(), block_sizes(model); groups = _group_labels(rows); n = len(rows); Q = max_len - k; cache = {}; rgb_t, jnt_t, anchor_t = [], [], []
    for row in rows:
        key = str(row.get("_source_id", row["id"]))
        if key not in cache:
            img = np.asarray(row["image"], np.float32); idx = np.minimum(np.arange(k, max_len), len(img) - 1); visual_target = cp.asarray(img[idx], cp.float32) - image_mean[None, :]; r_t = (visual_target @ basis).ravel(); jt = np.asarray(row["joints"], np.float32); q_anchor = jt[k - 1]; target_dq = (jt[idx] - q_anchor[None, :]) / qstd; j_t = cp.asarray(target_dq.ravel(), cp.float32); anchor_norm = (q_anchor - qmean) / qstd; a_t = cp.asarray(anchor_norm, cp.float32); cache[key] = (r_t, j_t, a_t)
        rgb_t.append(cache[key][0]); jnt_t.append(cache[key][1]); anchor_t.append(cache[key][2])
    Yrgb = cp.stack(rgb_t); rgb_fit = _fit_ridge_matrixfree(raw, Yrgb, groups, sizes, weights, grid, int(model["seed"]) + 1009 * int(k) + 11)
    head = dict(fmean=rgb_fit["fmean"], fscale=rgb_fit["fscale"], rgb_W=rgb_fit["W"], rgb_b=rgb_fit["b"], image_mean=image_mean.astype(cp.float32), basis=basis, q_values=np.arange(k, max_len, dtype=np.int32), Q=int(Q), class_heads={})
    X = rgb_fit["X"]; info = dict(n=n, groups=int(len(np.unique(groups))), lam_rgb=rgb_fit["lam"], ridge_rel_rgb=rgb_fit["ridge_rel"], rmse_rgb=rgb_fit["cv_rmse"], base_rgb=float(cp.sqrt(cp.mean((Yrgb - Yrgb.mean(axis=0)) ** 2))), solver_rgb=rgb_fit["solver"], cg_rgb=rgb_fit["cg_iters"], resid_rgb=rgb_fit["cg_resid"])
    Yj = cp.stack(jnt_t); rel_j, cv_j = _select_ridge_relative_lambda(raw, Yj, groups, sizes, weights, grid, int(model["seed"]) + 1009 * int(k) + 23); jnt_W, jnt_b, lam_j, sol_j, it_j, res_j = _ridge_matrixfree_fit(X, Yj, rel_j); head["jnt_W"] = jnt_W.astype(cp.float32); head["jnt_b"] = jnt_b.astype(cp.float32); info.update(lam_jnt=lam_j, ridge_rel_jnt=rel_j, rmse_jnt=cv_j, base_jnt=float(cp.sqrt(cp.mean((Yj - Yj.mean(axis=0)) ** 2))), solver_jnt=sol_j, cg_jnt=it_j, resid_jnt=res_j)
    Ya = cp.stack(anchor_t); vision_ix = np.asarray([i for i, row in enumerate(rows) if _row_mask(row, model)[1] == 0.0], dtype=np.int32)
    if len(vision_ix) == 0:
        raise ValueError("Delta-q v11 requires vision-only dropout rows to train the q(K-1) anchor.")
    vision_cp = cp.asarray(vision_ix, dtype=cp.int32); raw_v = raw[vision_cp]; X_v = X[vision_cp]; Ya_v = Ya[vision_cp]; groups_v = groups[vision_ix]; rel_a, cv_a = _select_ridge_relative_lambda(raw_v, Ya_v, groups_v, sizes, weights, grid, int(model["seed"]) + 1009 * int(k) + 31); anchor_W, anchor_b, lam_a, sol_a, it_a, res_a = _ridge_matrixfree_fit(X_v, Ya_v, rel_a); head["anchor_W"] = anchor_W.astype(cp.float32); head["anchor_b"] = anchor_b.astype(cp.float32); info.update(lam_anchor=lam_a, ridge_rel_anchor=rel_a, rmse_anchor=cv_a, solver_anchor=sol_a, cg_anchor=it_a, resid_anchor=res_a)
    del Yj, Ya, jnt_W, anchor_W, raw_v, X_v, Ya_v
    if with_cls:
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.float32)
        labels = cp.asarray(labels[:, None])
        rel_c, cv_c = _select_ridge_relative_lambda(raw, labels, groups, sizes, weights, grid, int(model["seed"]) + 1009 * int(k) + 37); cls_W, cls_b, lam_c, sol_c, it_c, res_c = _ridge_matrixfree_fit(X, labels, rel_c); pred = X @ cls_W + cls_b; info.update(lam_cls=lam_c, ridge_rel_cls=rel_c, grouped_cls_rmse=cv_c, train_acc=float(cp.mean((pred >= 0.5) == (labels >= 0.5))), solver_cls=sol_c, cg_cls=it_c, resid_cls=res_c); head["cls_W"] = cls_W.astype(cp.float32); head["cls_b"] = cls_b.astype(cp.float32); del labels, cls_W
    del X, Yrgb; cp.get_default_memory_pool().free_all_blocks(); return head, info


def _fmt(info):
    out = (f"n={info['n']}; groups={info.get('groups', info['n'])}; rgb CV={info['rmse_rgb']:.3f} (base={info['base_rgb']:.3f}, rel={info['ridge_rel_rgb']:.1e}, {info['solver_rgb']}, cg={info['cg_rgb']}, res={info['resid_rgb']:.1e})")
    if "rmse_jnt" in info:
        out += (f"; delta-joints CV={info['rmse_jnt']:.3f} (base={info['base_jnt']:.3f}, rel={info['ridge_rel_jnt']:.1e}, {info['solver_jnt']}, cg={info['cg_jnt']}, res={info['resid_jnt']:.1e})")
    if "rmse_anchor" in info:
        out += (f"; visual-anchor CV={info['rmse_anchor']:.3f} (rel={info['ridge_rel_anchor']:.1e}, {info['solver_anchor']}, cg={info['cg_anchor']}, res={info['resid_anchor']:.1e})")
    return out


def build_model(train, N, seed=42):
    from data_utils import compute_train_stats
    for row in train:
        _validate(row)
    all_stats = compute_train_stats(train); stats = {"qmean": np.asarray(all_stats["qmean"], np.float32), "qstd": np.asarray(all_stats["qstd"], np.float32)}
    model = new_model(int(N), int(seed), stats); rank = int(getattr(p, "IMAGE_PCA_RANK", 48)); basis, image_mean = _image_basis(train, rank, seed, absolute=True); model["bases"]["all"] = basis; model["image_means"]["all"] = image_mean; model["basis"] = basis; classes = sorted({int(row["label"]) for row in train}); min_rows = int(getattr(p, "MIN_CLASS_ROWS", 8)); cond = True
    if cond:
        for c in classes:
            rows_c = [row for row in train if int(row["label"]) == c]
            if len(rows_c) >= min_rows:
                basis_c, mean_c = _image_basis(rows_c, rank, seed, absolute=True); model["bases"][c] = basis_c; model["image_means"][c] = mean_c
    dropout_rows = _expand_dropout_rows(train); specs = _training_specs(dropout_rows); model["dropout_pattern"] = np.asarray([(name, count) for name, count in _dropout_pattern()], dtype="U16"); model["training_noise_enabled"] = bool(getattr(p, "TRAIN_SENSOR_NOISE", False)); model["training_noise_levels"] = np.asarray(_training_noise_levels() if model["training_noise_enabled"] else (), dtype=np.float32); model["training_noise_copies"] = (int(getattr(p, "TRAIN_NOISE_COPIES", 1)) if model["training_noise_enabled"] else 0)
    print(f"MODALITY DROPOUT: {len(train)} clean trajectories -> {len(dropout_rows)} dropout observations; pattern={_dropout_pattern()}; grouped Ridge folds={getattr(p, 'RIDGE_GROUP_FOLDS', 5)}")
    if model["training_noise_enabled"]:
        print(f"TRAIN SENSOR NOISE (lazy): {len(dropout_rows)} -> {len(specs)} observations; levels={tuple(float(x) for x in model['training_noise_levels'])}; copies={model['training_noise_copies']}; conditions={_training_noise_conditions()}; clean targets=True")
    else:
        print("TRAIN SENSOR NOISE: disabled")
    joint_target = "delta_from_qK_with_visual_anchor"
    print(f"PCA de imagen: {model['basis'].shape[1]} componentes; clases={classes}; condicionado=True; vision_input_scale={model['vision_input_scale']}; joint_target={joint_target}; solver=matrixfree-ridge")
    print(f"MATRIXFREE: CV_max_rows={getattr(p, 'RIDGE_CV_MAX_ROWS', 1200)}; CV_outputs={getattr(p, 'RIDGE_CV_OUTPUTS', 64)}; CG_tol={getattr(p, 'RIDGE_CG_TOL', 1e-4)}; CG_max_iter={getattr(p, 'RIDGE_CG_MAX_ITER', 80)}; output_chunk={getattr(p, 'RIDGE_OUTPUT_CHUNK', 128)}")
    for k in p.READOUT_K:
        k = int(k); idx, raw = _training_features_for_k(specs, model, stats, seed, k)
        if raw is None:
            raise ValueError(f"No trajectories longer than K={k}")
        rows = [specs[i][0] for i in idx]; max_len = max(len(row["image"]) for row in rows); head, info = _fit_head(rows, raw, model, k, model["bases"]["all"], model["image_means"]["all"], max_len, with_cls=True); model["heads"][k] = head; model["accuracy"][k] = info["train_acc"]; print(f"K={k}: Q={max_len-k}; train acc={info['train_acc']:.3f}; grouped cls RMSE={info['grouped_cls_rmse']:.3f}; [general] {_fmt(info)}")
        if info.get("resid_rgb", 0.0) > 10.0 * float(getattr(p, "RIDGE_CG_TOL", 1e-4)):
            print(f"   WARNING RGB CG residual={info['resid_rgb']:.2e}; consider increasing RIDGE_CG_MAX_ITER.")
        if info.get("resid_jnt", 0.0) > 10.0 * float(getattr(p, "RIDGE_CG_TOL", 1e-4)):
            print(f"   WARNING JOINT CG residual={info['resid_jnt']:.2e}; consider increasing RIDGE_CG_MAX_ITER.")
        if cond:
            for c in classes:
                ix = [i for i, row in enumerate(rows) if int(row["label"]) == c]
                if len(ix) < min_rows or c not in model["bases"]:
                    print(f"   clase {c}: solo {len(ix)} observaciones, se usa la cabeza general")
                    continue
                ix_cp = cp.asarray(ix, dtype=cp.int32); sub, sinfo = _fit_head([rows[i] for i in ix], raw[ix_cp], model, k, model["bases"][c], model["image_means"][c], max_len, with_cls=False); head["class_heads"][c] = sub; print(f"   [clase {c}] {_fmt(sinfo)}")
        del raw, rows; cp.get_default_memory_pool().free_all_blocks()
    return model


def predict_trajectory(model, row, k, horizon=None, use_class=None, modality_mode=None):
    gen = model["heads"][int(k)]; Q = gen["Q"]; horizon = min(len(row["image"]) - k, Q) if horizon is None else int(horizon)
    if horizon < 0 or horizon > Q:
        raise ValueError("Invalid horizon")
    window = int(getattr(p, "SMOOTH_WINDOW", 3)); obs_row = dict(row); condition = modality_mode if modality_mode is not None else getattr(p, "MODALITY_TEST_MODE", "full")
    obs_row["_modality_mask"] = np.asarray(_dropout_mask(condition), np.float32); obs_row["_modality_condition"] = str(condition); raw = raw_feature(obs_row, model, k); score = float((_norm(raw, gen) @ gen["cls_W"] + gen["cls_b"]).item()); c = int(score >= 0.5) if use_class is None else int(use_class); head = gen["class_heads"].get(c, gen); x = _norm(raw, head); basis = head["basis"]; r = basis.shape[1]; coef = (x @ head["rgb_W"] + head["rgb_b"]).reshape(Q, r); coef = _smooth(coef, window)[:horizon]; base_img = head["image_mean"]; delta_img = cp.clip(coef @ basis.T, -p.DELTA_CLIP, p.DELTA_CLIP); images = cp.asnumpy(cp.clip(base_img[None, :] + delta_img, 0, 1)).astype(np.float32)
    dqp = (x @ head["jnt_W"] + head["jnt_b"]).reshape(Q, p.JOINT_DIM); dqp = _smooth(dqp, window)[:horizon]; qstd = cp.asarray(model["stats"]["qstd"], cp.float32); qmean = cp.asarray(model["stats"]["qmean"], cp.float32); _, joints_on = _row_mask(obs_row, model)
    if joints_on != 0.0:
        q_anchor = cp.asarray(_obs_joints(obs_row)[k - 1], cp.float32)
    else:
        anchor_norm = x @ head["anchor_W"] + head["anchor_b"]; q_anchor = qmean + anchor_norm * qstd
    joints = cp.asnumpy(q_anchor[None, :] + dqp * qstd).astype(np.float32)
    return images, joints, score


def _save_head(payload, tag, head):
    for name in HEAD_ARRAYS:
        payload[f"{name}_{tag}"] = cp.asnumpy(head[name])
    payload[f"jnt_W_{tag}"] = cp.asnumpy(head["jnt_W"]); payload[f"jnt_b_{tag}"] = cp.asnumpy(head["jnt_b"]); payload[f"qvals_{tag}"] = head["q_values"]


def _load_head(d, tag, basis):
    head = {name: cp.asarray(d[f"{name}_{tag}"], cp.float32) for name in HEAD_ARRAYS}; head["jnt_W"] = cp.asarray(d[f"jnt_W_{tag}"], cp.float32); head["jnt_b"] = cp.asarray(d[f"jnt_b_{tag}"], cp.float32); head["q_values"] = np.asarray(d[f"qvals_{tag}"], np.int32); head["Q"] = int(len(head["q_values"])); head["basis"] = basis; head["class_heads"] = {}
    return head


def save_model(path, model):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); payload = {key: cp.asnumpy(model[key]) for key in ("Win", "W", "bias")}
    payload["stats_qmean"] = np.asarray(model["stats"]["qmean"], np.float32); payload["stats_qstd"] = np.asarray(model["stats"]["qstd"], np.float32)
    for key, basis in model["bases"].items():
        payload[f"basis_{key}"] = cp.asnumpy(basis)
    for k, head in model["heads"].items():
        _save_head(payload, f"{k}", head); payload[f"cls_W_{k}"] = cp.asnumpy(head["cls_W"]); payload[f"cls_b_{k}"] = cp.asnumpy(head["cls_b"]); payload[f"classes_{k}"] = np.asarray(sorted(head["class_heads"]), np.int32)
        for c, sub in head["class_heads"].items():
            _save_head(payload, f"{k}_c{c}", sub)
    np.savez_compressed(path, **payload, N=model["N"], seed=model["seed"], mode="dropout", vision_input_scale=np.float32(model["vision_input_scale"]), training_pipeline=TRAINING_PIPELINE, ridge_solver="matrixfree_pcg", ridge_cv_max_rows=int(getattr(p, "RIDGE_CV_MAX_ROWS", 1200)), ridge_cv_outputs=int(getattr(p, "RIDGE_CV_OUTPUTS", 64)), ridge_cg_tol=np.float32(getattr(p, "RIDGE_CG_TOL", 1e-4)), ridge_cg_max_iter=int(getattr(p, "RIDGE_CG_MAX_ITER", 80)), training_noise_enabled=int(bool(model.get("training_noise_enabled", False))), training_noise_levels=np.asarray(model.get("training_noise_levels", ()), dtype=np.float32), training_noise_copies=int(model.get("training_noise_copies", 0)), joint_target_mode="delta_from_qK_with_visual_anchor", dropout_version=3, dropout_pattern=np.asarray(model.get("dropout_pattern", []), dtype="U16"), image_dim=model["image_dim"], ks=np.asarray(sorted(model["heads"]), np.int32), architecture=ARCHITECTURE)


def load_model(path):
    with np.load(path, allow_pickle=False) as d:
        architecture = str(d["architecture"])
        if architecture != ARCHITECTURE:
            raise ValueError(f"Modelo con arquitectura {architecture}; se espera {ARCHITECTURE}. Reentrena v11.")
        if str(d["mode"]) != "dropout":
            raise ValueError("Delta-q v11 only loads dropout models.")
        if "joint_target_mode" not in d.files or str(d["joint_target_mode"]) != "delta_from_qK_with_visual_anchor":
            raise ValueError("Model is not a delta-q v11 model.")
        if "dropout_version" not in d.files or int(d["dropout_version"]) != 3:
            raise ValueError("Delta-q v11 dropout metadata missing or incompatible.")
        vision_input_scale = float(d["vision_input_scale"])
        if not np.isclose(vision_input_scale, float(p.VISION_INPUT_SCALE)) or int(d["image_dim"]) != p.IMAGE_DIM:
            raise ValueError("Model/config vision-input-scale or image dimension mismatch.")
        stats = {"qmean": np.asarray(d["stats_qmean"], np.float32), "qstd": np.asarray(d["stats_qstd"], np.float32)}
        model = {name: cp.asarray(d[name], cp.float32) for name in ("Win", "W", "bias")}
        bases = {}
        for key in d.files:
            if key.startswith("basis_"):
                tag = key[len("basis_"):]; bases["all" if tag == "all" else int(tag)] = cp.asarray(d[key], cp.float32)
        model.update(N=int(d["N"]), seed=int(d["seed"]), mode="dropout", vision_input_scale=vision_input_scale, image_dim=int(d["image_dim"]), stats=stats, heads={}, accuracy={}, bases=bases, basis=bases["all"], image_means={}, dropout_pattern=(np.asarray(d["dropout_pattern"]) if "dropout_pattern" in d.files else np.empty((0, 2), dtype="U16")))
        for k in d["ks"]:
            k = int(k); head = _load_head(d, f"{k}", bases["all"]); head["cls_W"] = cp.asarray(d[f"cls_W_{k}"], cp.float32); head["cls_b"] = cp.asarray(d[f"cls_b_{k}"], cp.float32)
            for c in d[f"classes_{k}"]:
                c = int(c); head["class_heads"][c] = _load_head(d, f"{k}_c{c}", bases[c])
            model["heads"][k] = head
    return model
