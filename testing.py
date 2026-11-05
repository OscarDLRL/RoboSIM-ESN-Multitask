"""Evaluate the cleaned baseline ESN, optionally under sensor noise.

Robustness protocol:
- The model is NOT retrained.
- Only the first K observed frames are corrupted.
- Future RGB/joint targets remain clean.
- Metrics are always computed against the original clean trajectory.
"""

import argparse
import csv
import sys

import cupy as cp
import numpy as np

import metrics
import robustness
from model_loader import load_model_package


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        choices=("baseline","motion_only"),
        default="baseline",
    )

    ap.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )

    ap.add_argument("--neurons", type=int)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--save-predictions", type=int, default=2)

    ap.add_argument(
        "--rgb-noise",
        type=float,
        default=0.0,
        help="Uniform RGB noise epsilon in normalized [0,1] image units.",
    )
    ap.add_argument(
        "--joint-noise",
        type=float,
        default=0.0,
        help="Gaussian joint noise sigma in units of training qstd.",
    )
    ap.add_argument(
        "--noise-seed",
        type=int,
        default=123,
        help="Base random seed for sensor corruption.",
    )
    ap.add_argument(
        "--noise-repeats",
        type=int,
        default=1,
        help="Independent noise realizations per trajectory/K.",
    )

    a = ap.parse_args()

    if a.rgb_noise < 0:
        raise ValueError("--rgb-noise must be >= 0")

    if a.joint_noise < 0:
        raise ValueError("--joint-noise must be >= 0")

    if a.noise_repeats < 1:
        raise ValueError("--noise-repeats must be >= 1")

    p, esn = load_model_package(a.model)
    sys.modules["parameters"] = p
    sys.modules["esn"] = esn

    from data_utils import load_split

    n = a.neurons or p.NEURON_COUNTS[0]

    clean_data = load_split(
        a.split,
        data_path=p.DATA_PATH,
    )

    model_path = (
        p.MODEL_DIR
        / f"N_{n}"
        / f"run_{a.run:02d}"
        / "model.npz"
    )

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = esn.load_model(model_path)

    modality_mode = getattr(
        p,
        "MODALITY_TEST_MODE",
        "full",
    )

    corruption = robustness.robustness_tag(
        a.rgb_noise,
        a.joint_noise,
    )

    out = (
        p.TEST_RESULTS_DIR
        / a.split
        / "robustness"
        / corruption
        / f"N_{n}"
        / f"run_{a.run:02d}"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nMODEL={a.model}"
        f"\nARCH={esn.ARCHITECTURE}"
        f"\nDATA={p.DATA_PATH}"
        f"\nMODALITY={modality_mode}"
        f"\nRGB_NOISE={a.rgb_noise}"
        f"\nJOINT_NOISE={a.joint_noise} qstd"
        f"\nNOISE_REPEATS={a.noise_repeats}"
        f"\nOUTPUT={out}\n"
    )

    qstd = np.asarray(
        model["stats"]["qstd"],
        dtype=np.float32,
    )

    records = []

    for k in p.TEST_K:
        saved = {
            "pick": 0,
            "push": 0,
        }

        for clean_row in clean_data:
            if clean_row["length"] <= k:
                continue

            for repeat in range(a.noise_repeats):
                obs_row = robustness.corrupt_observation(
                    clean_row,
                    k,
                    qstd=qstd,
                    rgb_epsilon=a.rgb_noise,
                    joint_sigma=a.joint_noise,
                    seed=a.noise_seed,
                    repeat=repeat,
                )

                pred_img, pred_q, score = esn.predict_trajectory(
                    model,
                    obs_row,
                    k,
                    modality_mode=modality_mode,
                )

                gt = np.asarray(
                    clean_row["image"][
                        k : k + len(pred_img)
                    ],
                    dtype=np.float32,
                )

                if not len(gt) or len(gt) != len(pred_img):
                    continue

                prev_clean = np.asarray(
                    clean_row["image"][k - 1],
                    dtype=np.float32,
                )

                prev_observed = np.asarray(
                    obs_row["image"][k - 1],
                    dtype=np.float32,
                )

                visual = metrics.prediction_metrics(
                    gt,
                    pred_img,
                    prev_clean,
                    p.IMAGE_HEIGHT,
                    p.IMAGE_WIDTH,
                    getattr(
                        p,
                        "MOTION_THRESHOLD",
                        0.08,
                    ),
                )

                rec = dict(
                    split=a.split,
                    task=clean_row["kind"],
                    trajectory=clean_row["id"],
                    K=int(k),
                    repeat=int(repeat),
                    frames=len(pred_img),
                    modality=modality_mode,
                    rgb_noise=float(a.rgb_noise),
                    joint_noise=float(a.joint_noise),
                    push_probability=float(score),
                    true_label=int(clean_row["label"]),
                    predicted_label=int(score >= 0.5),
                    correct=int(
                        int(score >= 0.5)
                        == int(clean_row["label"])
                    ),
                    **visual,
                )

                if pred_q is not None:
                    qtrue = np.asarray(
                        clean_row["joints"][
                            k : k + len(pred_q)
                        ],
                        dtype=np.float32,
                    )

                    qpred = np.asarray(
                        pred_q,
                        dtype=np.float32,
                    )

                    if qtrue.shape != qpred.shape:
                        raise ValueError(
                            f"{clean_row['id']}: "
                            f"joints {qtrue.shape} != {qpred.shape}"
                        )

                    rec["joint_mae"] = float(
                        np.mean(
                            np.abs(
                                qpred - qtrue
                            )
                        )
                    )

                    rec["joint_rmse"] = float(
                        np.sqrt(
                            np.mean(
                                (qpred - qtrue) ** 2
                            )
                        )
                    )

                else:
                    qtrue = np.empty(
                        (0, 9),
                        dtype=np.float32,
                    )

                    qpred = np.empty(
                        (0, 9),
                        dtype=np.float32,
                    )

                    rec["joint_mae"] = np.nan
                    rec["joint_rmse"] = np.nan

                records.append(rec)

                if saved[clean_row["kind"]] < a.save_predictions:
                    folder = (
                        out
                        / f"K_{k}"
                        / "predictions"
                    )

                    folder.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    np.savez_compressed(
                        folder
                        / (
                            f"{clean_row['id']}"
                            f"_r{repeat:02d}.npz"
                        ),
                        clean_previous_image=prev_clean,
                        observed_previous_image=prev_observed,
                        observed_image_prefix=np.asarray(
                            obs_row["image"][:k],
                            dtype=np.float32,
                        ),
                        observed_joint_prefix=np.asarray(
                            obs_row["joints"][:k],
                            dtype=np.float32,
                        ),
                        true_image=gt,
                        pred_image=np.asarray(
                            pred_img,
                            dtype=np.float32,
                        ),
                        true_joints=qtrue,
                        pred_joints=qpred,
                        goal=clean_row["goal"],
                        action=clean_row["kind"],
                        K=int(k),
                        repeat=int(repeat),
                        modality=modality_mode,
                        rgb_noise=float(a.rgb_noise),
                        joint_noise=float(a.joint_noise),
                    )

                    saved[clean_row["kind"]] += 1

        sub = [
            r
            for r in records
            if r["K"] == k
        ]

        if sub:
            print(
                f"K={k}: "
                f"n={len(sub)} "
                f"acc={np.mean([r['correct'] for r in sub]):.3f} "
                f"pixel={np.mean([r['pixel_rmse'] for r in sub]):.4f} "
                f"delta={np.mean([r['delta_image_rmse'] for r in sub]):.4f} "
                f"joint={np.nanmean([r['joint_rmse'] for r in sub]):.4f}"
            )

    if not records:
        raise ValueError(
            "No predictions generated"
        )

    path = out / "metrics.csv"

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0]),
        )

        writer.writeheader()
        writer.writerows(records)

    print("Saved:", path)

    del model
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()