"""Direct future-frame delta ESN with oracle future-joint RGB conditioning.

Diagnostic experiment:
- Separate Pick/Push heads (true label selects the head).
- Joint forecast uses only the first K observed frames.
- RGB forecast additionally receives the TRUE future joint configuration at q.
  This is intentional leakage used only to test whether future pose information
  is the missing ingredient for visual reconstruction.
"""
from pathlib import Path
import numpy as np
import cupy as cp
import parameters as p


def configuration():
    mode = p.EXPERIMENT_MODE
    if mode not in ("vision", "multimodal"):
        raise ValueError("Choose vision or multimodal")
    return mode, bool(p.USE_GOAL)


def input_width():
    mode, goal = configuration()
    return (
        p.IMAGE_DIM
        + (p.JOINT_DIM if mode == "multimodal" else 0)
        + (p.GOAL_DIM if goal else 0)
    )


def _validate(row):
    image = np.asarray(row["image"])
    if image.ndim != 2 or image.shape[1] != p.IMAGE_DIM:
        raise ValueError(
            f"{row['id']}: expected image (T,{p.IMAGE_DIM}), got {image.shape}"
        )
    if len(image) != len(row["joints"]):
        raise ValueError(f"{row['id']}: image/joint frame mismatch")
    if np.asarray(row["goal"]).shape != (3,):
        raise ValueError(f"{row['id']}: expected goal XYZ shape (3,)")
    label = int(row["label"])
    if label not in (0, 1):
        raise ValueError(
            f"{row['id']}: expected label 0=Pick or 1=Push, got {label}"
        )
    return image


def new_model(N, seed, stats):
    rng = cp.random.RandomState(int(seed))
    dim = input_width()

    Win = rng.uniform(
        -p.INPUT_SCALE, p.INPUT_SCALE, (N, dim)
    ).astype(cp.float32)

    bias = rng.uniform(
        -p.INPUT_SCALE, p.INPUT_SCALE, (N,)
    ).astype(cp.float32)

    W = rng.uniform(-1, 1, (N, N)).astype(cp.float32)

    radius = float(
        cp.max(cp.abs(cp.linalg.eigvals(W))).get()
    )
    W *= cp.float32(
        p.SPECTRAL_RADIUS / max(radius, 1e-12)
    )

    return dict(
        Win=Win,
        W=W,
        bias=bias,
        N=N,
        seed=int(seed),
        mode=p.EXPERIMENT_MODE,
        use_goal=bool(p.USE_GOAL),
        image_dim=p.IMAGE_DIM,
        stats=stats,
        heads={},
        accuracy={},
    )


def prepare_observation(row, model, k):
    """One input vector per observed transition t=1..K-1."""
    image = cp.asarray(row["image"][:k], cp.float32)
    modalities = [image[1:] - image[:-1]]

    if model["mode"] == "multimodal":
        raw_q = cp.asarray(row["joints"][:k], cp.float32)
        mean = cp.asarray(model["stats"]["qmean"], cp.float32)
        std = cp.asarray(model["stats"]["qstd"], cp.float32)
        q = (raw_q - mean) / std
        modalities.append(
            cp.float32(p.JOINT_INPUT_SCALE) * (q[1:] - q[:-1])
        )

    if model["use_goal"]:
        goal = (
            cp.asarray(row["goal"], cp.float32)
            - cp.asarray(model["stats"]["gmean"], cp.float32)
        ) / cp.asarray(model["stats"]["gstd"], cp.float32)

        modalities.append(
            cp.broadcast_to(
                cp.float32(p.GOAL_INPUT_SCALE) * goal,
                (k - 1, p.GOAL_DIM),
            )
        )

    return cp.concatenate(modalities, axis=1)


def feature(row, model, k):
    obs = prepare_observation(row, model, k)

    state = cp.zeros(
        (model["N"],), dtype=cp.float32
    )

    for t in range(len(obs)):
        cand = cp.tanh(
            model["bias"]
            + model["Win"] @ obs[t]
            + model["W"] @ state
        )
        state = (
            (1 - p.LEAK_RATE) * state
            + p.LEAK_RATE * cand
        )

    c = p.IMAGE_CHANNELS
    h, w = p.IMAGE_HEIGHT, p.IMAGE_WIDTH
    ah, aw = p.APPEARANCE_HEIGHT, p.APPEARANCE_WIDTH

    img = cp.asarray(
        row["image"][k - 1], cp.float32
    ).reshape(h, w, c)

    small = (
        img.reshape(
            ah, h // ah,
            aw, w // aw,
            c
        )
        .mean(axis=(1, 3))
        .ravel()
        - cp.float32(0.5)
    )

    base = [
        cp.ones((1,), dtype=cp.float32),
        state,
        small,
    ]

    if model["mode"] == "multimodal":
        q = (
            cp.asarray(row["joints"][k - 1], cp.float32)
            - cp.asarray(model["stats"]["qmean"], cp.float32)
        ) / cp.asarray(model["stats"]["qstd"], cp.float32)

        base.append(
            cp.float32(p.JOINT_INPUT_SCALE) * q
        )

    if model["use_goal"]:
        goal = (
            cp.asarray(row["goal"], cp.float32)
            - cp.asarray(model["stats"]["gmean"], cp.float32)
        ) / cp.asarray(model["stats"]["gstd"], cp.float32)

        base.append(
            cp.float32(p.GOAL_INPUT_SCALE) * goal
        )

    return cp.concatenate(base)


def _rgb_target(row, k, q):
    """Cumulative RGB change from last observed frame K-1 to future q."""
    return np.asarray(
        row["image"][q] - row["image"][k - 1],
        np.float32,
    )


def _joint_target(row, k, q, model):
    """Normalized cumulative joint change from K-1 to future q."""
    qstd = model["stats"]["qstd"]
    return np.asarray(
        (row["joints"][q] - row["joints"][k - 1]) / qstd,
        np.float32,
    )


def _future_joint_feature(row, q, model):
    """TRUE future normalized joint configuration, used only by RGB oracle."""
    qmean = model["stats"]["qmean"]
    qstd = model["stats"]["qstd"]

    future_q = (
        np.asarray(row["joints"][q], np.float32) - qmean
    ) / qstd

    return (
        np.float32(p.FUTURE_JOINT_SCALE)
        * future_q.astype(np.float32)
    )


def build_model(
    train,
    N,
    training_epsilon=0,
    augmentation_count=0,
    seed=42,
):
    if training_epsilon != 0 or augmentation_count:
        raise NotImplementedError(
            "This version deliberately disables augmentation/noise."
        )

    if p.EXPERIMENT_MODE != "multimodal":
        raise ValueError(
            'oracle_future_q_rgb requires EXPERIMENT_MODE="multimodal"'
        )

    from data_utils import compute_train_stats

    stats = compute_train_stats(train)
    model = new_model(int(N), int(seed), stats)

    for row in train:
        _validate(row)

    for k in p.READOUT_K:
        rows = [
            r for r in train
            if len(r["image"]) > k
        ]

        if not rows:
            raise ValueError(
                f"No trajectories longer than K={k}"
            )

        n_pick = sum(
            int(r["label"]) == 0 for r in rows
        )
        n_push = sum(
            int(r["label"]) == 1 for r in rows
        )

        print(
            f"\nK={k}: Pick={n_pick}, "
            f"Push={n_push}, Total={len(rows)}"
        )

        X = cp.stack(
            [feature(row, model, k) for row in rows]
        ).astype(cp.float32)

        # Shared Pick/Push classifier: diagnostic only.
        cls_y = cp.asarray(
            np.array(
                [r["label"] for r in rows],
                np.float64,
            )[:, None]
        )

        X64 = X.astype(cp.float64)
        gram = (
            X64 @ X64.T
            + p.READOUT_RIDGE
            * cp.eye(len(rows), dtype=cp.float64)
        )

        cls_alpha = cp.linalg.solve(
            gram, cls_y
        ).astype(cp.float32)

        fitted_scores = (
            (X @ X.T) @ cls_alpha
        ).ravel().get()

        fitted_labels = fitted_scores >= 0.5

        true_labels = np.asarray(
            [int(r["label"]) for r in rows]
        )

        cls_accuracy = float(
            np.mean(fitted_labels == true_labels)
        )

        max_len = max(
            len(row["image"]) for row in rows
        )

        joint_alphas = {0: [], 1: []}
        rgb_alphas = {0: [], 1: []}
        indices = {0: [], 1: []}
        future_q_train = {0: [], 1: []}

        for q in range(k, max_len):
            for label in (0, 1):
                ix = [
                    j
                    for j, row in enumerate(rows)
                    if (
                        len(row["image"]) > q
                        and int(row["label"]) == label
                    )
                ]

                if not ix:
                    joint_alphas[label].append(None)
                    rgb_alphas[label].append(None)
                    indices[label].append(None)
                    future_q_train[label].append(None)
                    continue

                Xq = X[ix].astype(cp.float64)

                # ------------------------------
                # Joint forecast: X -> delta q
                # ------------------------------
                Y_joint = cp.asarray(
                    np.stack(
                        [
                            _joint_target(
                                rows[j], k, q, model
                            )
                            for j in ix
                        ]
                    ),
                    cp.float64,
                )

                A_joint = (
                    Xq @ Xq.T
                    + p.READOUT_RIDGE
                    * cp.eye(
                        len(ix), dtype=cp.float64
                    )
                )

                joint_alpha = cp.linalg.solve(
                    A_joint, Y_joint
                ).astype(cp.float32)

                # --------------------------------------------------
                # RGB oracle: [X, TRUE future normalized q] -> delta RGB
                # --------------------------------------------------
                Qq = cp.asarray(
                    np.stack(
                        [
                            _future_joint_feature(
                                rows[j], q, model
                            )
                            for j in ix
                        ]
                    ),
                    cp.float64,
                )

                Y_rgb = cp.asarray(
                    np.stack(
                        [
                            _rgb_target(
                                rows[j], k, q
                            )
                            for j in ix
                        ]
                    ),
                    cp.float64,
                )

                # For Z=[Xq,Qq], Z Z^T = Xq Xq^T + Qq Qq^T.
                A_rgb = (
                    Xq @ Xq.T
                    + Qq @ Qq.T
                    + p.READOUT_RIDGE
                    * cp.eye(
                        len(ix), dtype=cp.float64
                    )
                )

                rgb_alpha = cp.linalg.solve(
                    A_rgb, Y_rgb
                ).astype(cp.float32)

                joint_alphas[label].append(
                    joint_alpha
                )
                rgb_alphas[label].append(
                    rgb_alpha
                )
                indices[label].append(
                    np.asarray(ix, dtype=np.int32)
                )
                future_q_train[label].append(
                    Qq.astype(cp.float32)
                )

        model["heads"][int(k)] = dict(
            X=X,
            cls_alpha=cls_alpha,
            q_values=np.arange(
                k, max_len, dtype=np.int32
            ),
            joint_alphas=joint_alphas,
            rgb_alphas=rgb_alphas,
            indices=indices,
            future_q_train=future_q_train,
        )

        model["accuracy"][int(k)] = cls_accuracy

        print(
            f"K={k}: n={len(rows)}; "
            f"final frame index={max_len - 1}; "
            f"training action accuracy={cls_accuracy:.3f}"
        )

    return model


def predict_trajectory(
    model,
    row,
    k,
    horizon=None,
    verbose=False,
):
    """Oracle future-q RGB diagnostic."""

    head = model["heads"][int(k)]
    max_h = len(head["q_values"])

    if horizon is None:
        horizon = min(
            len(row["image"]) - k,
            max_h,
        )
    else:
        horizon = int(horizon)

    if horizon < 0 or horizon > max_h:
        raise ValueError("Invalid horizon")

    x = feature(row, model, k)

    score = float(
        (
            x
            @ head["X"].T
            @ head["cls_alpha"]
        ).item()
    )

    predicted_label = int(score >= 0.5)

    # Oracle skill selection for this diagnostic.
    label = int(row["label"])
    if label not in (0, 1):
        raise ValueError(
            f"Unexpected label {label}; expected 0=Pick or 1=Push"
        )

    if verbose:
        true_name = (
            "Pick" if label == 0 else "Push"
        )
        predicted_name = (
            "Pick"
            if predicted_label == 0
            else "Push"
        )
        print(
            f"K={k} | TRUE={true_name} | "
            f"classifier score={score:.3f} | "
            f"classifier says={predicted_name}"
        )

    base_img = cp.asarray(
        row["image"][k - 1], cp.float32
    )
    base_q = cp.asarray(
        row["joints"][k - 1], cp.float32
    )

    outputs_img = []
    outputs_q = []

    qmean = cp.asarray(
        model["stats"]["qmean"], cp.float32
    )
    qstd = cp.asarray(
        model["stats"]["qstd"], cp.float32
    )

    for step in range(horizon):
        q = int(head["q_values"][step])

        ix = head["indices"][label][step]
        joint_alpha = (
            head["joint_alphas"][label][step]
        )
        rgb_alpha = (
            head["rgb_alphas"][label][step]
        )
        Qtrain = (
            head["future_q_train"][label][step]
        )

        if (
            ix is None
            or joint_alpha is None
            or rgb_alpha is None
            or Qtrain is None
        ):
            if verbose:
                print(
                    f"Stopping at q={q}: "
                    f"no training examples for label={label}"
                )
            break

        # ------------------------------------
        # Joints: legitimate forecast from x.
        # ------------------------------------
        common_kernel = (
            x @ head["X"][ix].T
        )

        joint_delta = (
            common_kernel @ joint_alpha
        )

        pred_q = (
            base_q + joint_delta * qstd
        )
        outputs_q.append(pred_q)

        # ------------------------------------
        # RGB: intentional oracle leakage.
        # ------------------------------------
        true_future_q = cp.asarray(
            row["joints"][q], cp.float32
        )

        q_condition = (
            (true_future_q - qmean) / qstd
        )

        q_condition = (
            cp.float32(p.FUTURE_JOINT_SCALE)
            * q_condition
        )

        rgb_kernel = (
            common_kernel
            + q_condition @ Qtrain.T
        )

        rgb_delta = (
            rgb_kernel @ rgb_alpha
        )

        rgb_delta = cp.clip(
            rgb_delta,
            -p.DELTA_CLIP,
            p.DELTA_CLIP,
        )

        pred_img = cp.clip(
            base_img + rgb_delta,
            0,
            1,
        )

        outputs_img.append(pred_img)

    images = (
        cp.asnumpy(cp.stack(outputs_img))
        if outputs_img
        else np.empty(
            (0, p.IMAGE_DIM),
            np.float32,
        )
    )

    joints = (
        cp.asnumpy(cp.stack(outputs_q))
        if outputs_q
        else np.empty(
            (0, p.JOINT_DIM),
            np.float32,
        )
    )

    return images, joints, score


def save_model(path, model):
    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    payload = {
        key: cp.asnumpy(model[key])
        for key in ("Win", "W", "bias")
    }

    for name, values in model["stats"].items():
        payload[f"stats_{name}"] = np.asarray(
            values
        )

    for k, head in model["heads"].items():
        payload[f"X_{k}"] = cp.asnumpy(
            head["X"]
        )
        payload[f"cls_{k}"] = cp.asnumpy(
            head["cls_alpha"]
        )
        payload[f"qvals_{k}"] = head[
            "q_values"
        ]

        for label in (0, 1):
            available = []

            for i in range(
                len(head["q_values"])
            ):
                joint_alpha = (
                    head["joint_alphas"][label][i]
                )
                rgb_alpha = (
                    head["rgb_alphas"][label][i]
                )
                ix = (
                    head["indices"][label][i]
                )
                Qtrain = (
                    head["future_q_train"][label][i]
                )

                exists = (
                    joint_alpha is not None
                    and rgb_alpha is not None
                    and ix is not None
                    and Qtrain is not None
                )

                available.append(int(exists))

                if not exists:
                    continue

                payload[
                    f"joint_a_{k}_{label}_{i}"
                ] = cp.asnumpy(joint_alpha)

                payload[
                    f"rgb_a_{k}_{label}_{i}"
                ] = cp.asnumpy(rgb_alpha)

                payload[
                    f"ix_{k}_{label}_{i}"
                ] = np.asarray(
                    ix, dtype=np.int32
                )

                payload[
                    f"future_q_{k}_{label}_{i}"
                ] = cp.asnumpy(Qtrain)

            payload[
                f"available_{k}_{label}"
            ] = np.asarray(
                available, dtype=np.uint8
            )

    for k, value in model.get(
        "accuracy", {}
    ).items():
        payload[f"accuracy_{k}"] = np.asarray(
            value, dtype=np.float32
        )

    np.savez_compressed(
        path,
        **payload,
        N=model["N"],
        seed=model["seed"],
        mode=model["mode"],
        use_goal=int(model["use_goal"]),
        image_dim=model["image_dim"],
        ks=np.asarray(
            sorted(model["heads"]),
            np.int32,
        ),
        architecture="maniskill_oracle_future_q_rgb_v1",
    )


def load_model(path):
    with np.load(
        path, allow_pickle=False
    ) as d:
        mode = str(d["mode"].item())
        use_goal = bool(int(d["use_goal"]))
        image_dim = int(d["image_dim"])

        if (
            mode != p.EXPERIMENT_MODE
            or use_goal != p.USE_GOAL
            or image_dim != p.IMAGE_DIM
        ):
            raise ValueError(
                "Model/config mode, goal, "
                "or image dimension mismatch"
            )

        stats = {
            name: np.asarray(
                d[f"stats_{name}"],
                np.float32,
            )
            for name in (
                "qmean",
                "qstd",
                "gmean",
                "gstd",
            )
        }

        model = {
            name: cp.asarray(
                d[name], cp.float32
            )
            for name in (
                "Win", "W", "bias"
            )
        }

        model.update(
            N=int(d["N"]),
            seed=int(d["seed"]),
            mode=mode,
            use_goal=use_goal,
            image_dim=image_dim,
            stats=stats,
            heads={},
            accuracy={},
        )

        for k_raw in d["ks"]:
            k = int(k_raw)

            qvals = np.asarray(
                d[f"qvals_{k}"],
                np.int32,
            )

            X = cp.asarray(
                d[f"X_{k}"],
                cp.float32,
            )

            cls_alpha = cp.asarray(
                d[f"cls_{k}"],
                cp.float32,
            )

            joint_alphas = {
                0: [], 1: []
            }
            rgb_alphas = {
                0: [], 1: []
            }
            indices = {
                0: [], 1: []
            }
            future_q_train = {
                0: [], 1: []
            }

            for label in (0, 1):
                available = np.asarray(
                    d[f"available_{k}_{label}"],
                    np.uint8,
                )

                if len(available) != len(
                    qvals
                ):
                    raise ValueError(
                        f"K={k}, label={label}: "
                        "availability length mismatch"
                    )

                for i, exists in enumerate(
                    available
                ):
                    if not exists:
                        joint_alphas[
                            label
                        ].append(None)
                        rgb_alphas[
                            label
                        ].append(None)
                        indices[
                            label
                        ].append(None)
                        future_q_train[
                            label
                        ].append(None)
                        continue

                    joint_alphas[
                        label
                    ].append(
                        cp.asarray(
                            d[
                                f"joint_a_{k}_{label}_{i}"
                            ],
                            cp.float32,
                        )
                    )

                    rgb_alphas[
                        label
                    ].append(
                        cp.asarray(
                            d[
                                f"rgb_a_{k}_{label}_{i}"
                            ],
                            cp.float32,
                        )
                    )

                    indices[
                        label
                    ].append(
                        np.asarray(
                            d[
                                f"ix_{k}_{label}_{i}"
                            ],
                            np.int32,
                        )
                    )

                    future_q_train[
                        label
                    ].append(
                        cp.asarray(
                            d[
                                f"future_q_{k}_{label}_{i}"
                            ],
                            cp.float32,
                        )
                    )

            model["heads"][k] = dict(
                X=X,
                cls_alpha=cls_alpha,
                q_values=qvals,
                joint_alphas=joint_alphas,
                rgb_alphas=rgb_alphas,
                indices=indices,
                future_q_train=future_q_train,
            )

            accuracy_key = f"accuracy_{k}"
            if accuracy_key in d.files:
                model["accuracy"][k] = float(
                    d[accuracy_key]
                )

    return model