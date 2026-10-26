#!/usr/bin/env python3
"""Evaluate generalized agent/interaction mask predictions from the object-structured ESN.

This script leaves the existing testing.py/reporting.py untouched.

Reports:
- binary IoU and F1 at parameters.MASK_THRESHOLD
- soft IoU (uses continuous mask prediction in [0,1])
- binary centroid error in mask pixels
- soft centroid error in mask pixels
- predicted/true mask mass

Important:
The interaction object can be very small. Soft metrics help distinguish:
    "no spatial signal"
from
    "correct location but low Ridge amplitude".
"""

import csv
from pathlib import Path

import numpy as np

import parameters as p
import data_utils as d

try:
    import esn_test as esn
except ImportError:
    import esn


EPS = 1e-8


def binary_iou(pred, true):
    pred = np.asarray(pred, dtype=bool)
    true = np.asarray(true, dtype=bool)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return 1.0 if union == 0 else float(inter / union)


def binary_f1(pred, true):
    pred = np.asarray(pred, dtype=bool)
    true = np.asarray(true, dtype=bool)
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    den = 2 * tp + fp + fn
    return 1.0 if den == 0 else float((2 * tp) / den)


def soft_iou(pred, true):
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    true = np.asarray(true, dtype=np.float64)
    inter = np.sum(pred * true)
    union = np.sum(pred + true - pred * true)
    return 1.0 if union <= EPS else float(inter / union)


def centroid(mask, weighted=False):
    arr = np.asarray(mask, dtype=np.float64)
    if not weighted:
        arr = (arr > 0).astype(np.float64)

    mass = arr.sum()
    if mass <= EPS:
        return None

    yy, xx = np.indices(arr.shape, dtype=np.float64)
    return np.array(
        [
            (yy * arr).sum() / mass,
            (xx * arr).sum() / mass,
        ],
        dtype=np.float64,
    )


def centroid_error(pred, true, weighted_pred=False):
    cpred = centroid(pred, weighted=weighted_pred)
    ctrue = centroid(true, weighted=False)

    if ctrue is None:
        return 0.0 if cpred is None else float(np.hypot(*true.shape))

    if cpred is None:
        # Maximum-scale penalty for an empty prediction.
        return float(np.hypot(*true.shape))

    return float(np.linalg.norm(cpred - ctrue))


def frame_metrics(soft_pred, true, threshold):
    soft_pred = np.clip(np.asarray(soft_pred, np.float32), 0.0, 1.0)
    true = (np.asarray(true) > 0).astype(np.uint8)
    binary_pred = soft_pred >= threshold

    return {
        "iou": binary_iou(binary_pred, true),
        "f1": binary_f1(binary_pred, true),
        "soft_iou": soft_iou(soft_pred, true),
        "centroid_error": centroid_error(binary_pred, true, weighted_pred=False),
        "soft_centroid_error": centroid_error(soft_pred, true, weighted_pred=True),
        "pred_mass": float(soft_pred.sum()),
        "true_mass": float(true.sum()),
        "pred_peak": float(soft_pred.max()) if soft_pred.size else 0.0,
        "binary_pixels": int(binary_pred.sum()),
    }


def mean_dict(items):
    keys = items[0].keys()
    return {k: float(np.mean([x[k] for x in items])) for k in keys}


def main():
    validation = d.load_split("validation", data_path=p.DATA_PATH)
    threshold = float(p.MASK_THRESHOLD)

    out_root = Path(p.TEST_RESULTS_DIR) / "mask_evaluation"
    out_root.mkdir(parents=True, exist_ok=True)

    detailed_rows = []

    for N in p.NEURON_COUNTS:
        for run in range(1, int(p.RUNS) + 1):
            model_path = (
                Path(p.MODEL_DIR)
                / f"N_{int(N)}"
                / f"run_{run:02d}"
                / "model.npz"
            )

            if not model_path.is_file():
                raise FileNotFoundError(model_path)

            model = esn.load_model(model_path)

            for k in p.TEST_K:
                rows = [r for r in validation if len(r["image"]) > k]

                for row in rows:
                    (
                        _pred_img,
                        _pred_q,
                        score,
                        pred_agent,
                        pred_interaction,
                    ) = esn.predict_trajectory(
                        model,
                        row,
                        int(k),
                        return_masks=True,
                        binary_masks=False,
                    )

                    horizon = len(pred_agent)

                    true_agent = (
                        esn._mask_sequence(row, "agent_mask")[k:k + horizon]
                        .reshape(
                            horizon,
                            model["mask_height"],
                            model["mask_width"],
                        )
                    )

                    true_interaction = (
                        esn._mask_sequence(row, "interaction_mask")[k:k + horizon]
                        .reshape(
                            horizon,
                            model["mask_height"],
                            model["mask_width"],
                        )
                    )

                    agent_metrics = [
                        frame_metrics(pred_agent[t], true_agent[t], threshold)
                        for t in range(horizon)
                    ]

                    interaction_metrics = [
                        frame_metrics(
                            pred_interaction[t],
                            true_interaction[t],
                            threshold,
                        )
                        for t in range(horizon)
                    ]

                    a = mean_dict(agent_metrics)
                    o = mean_dict(interaction_metrics)

                    detailed_rows.append(
                        {
                            "N": int(N),
                            "run": int(run),
                            "task": row["kind"],
                            "id": row["id"],
                            "K": int(k),
                            "frames_predicted": int(horizon),
                            "action_score": float(score),

                            "agent_iou": a["iou"],
                            "agent_f1": a["f1"],
                            "agent_soft_iou": a["soft_iou"],
                            "agent_centroid_error": a["centroid_error"],
                            "agent_soft_centroid_error": a["soft_centroid_error"],
                            "agent_pred_mass": a["pred_mass"],
                            "agent_true_mass": a["true_mass"],
                            "agent_pred_peak": a["pred_peak"],
                            "agent_binary_pixels": a["binary_pixels"],

                            "interaction_iou": o["iou"],
                            "interaction_f1": o["f1"],
                            "interaction_soft_iou": o["soft_iou"],
                            "interaction_centroid_error": o["centroid_error"],
                            "interaction_soft_centroid_error": o["soft_centroid_error"],
                            "interaction_pred_mass": o["pred_mass"],
                            "interaction_true_mass": o["true_mass"],
                            "interaction_pred_peak": o["pred_peak"],
                            "interaction_binary_pixels": o["binary_pixels"],
                        }
                    )

    detailed_path = out_root / "mask_metrics.csv"

    with detailed_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_rows[0].keys())
        writer.writeheader()
        writer.writerows(detailed_rows)

    # Aggregate by task/K.
    summary_rows = []
    for task in ("pick", "push"):
        for k in p.TEST_K:
            subset = [
                r
                for r in detailed_rows
                if r["task"] == task and int(r["K"]) == int(k)
            ]
            if not subset:
                continue

            fields = [
                "agent_iou",
                "agent_f1",
                "agent_soft_iou",
                "agent_centroid_error",
                "agent_soft_centroid_error",
                "agent_pred_mass",
                "agent_true_mass",
                "agent_pred_peak",
                "interaction_iou",
                "interaction_f1",
                "interaction_soft_iou",
                "interaction_centroid_error",
                "interaction_soft_centroid_error",
                "interaction_pred_mass",
                "interaction_true_mass",
                "interaction_pred_peak",
            ]

            row = {
                "task": task,
                "K": int(k),
                "n": len(subset),
            }

            for field in fields:
                row[field] = float(np.mean([x[field] for x in subset]))

            summary_rows.append(row)

    summary_path = out_root / "mask_summary.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nMASK SUMMARY @ threshold={threshold:.2f}\n")

    compact_fields = [
        "task",
        "K",
        "agent_iou",
        "agent_soft_iou",
        "agent_soft_centroid_error",
        "interaction_iou",
        "interaction_soft_iou",
        "interaction_soft_centroid_error",
        "interaction_pred_peak",
    ]

    header = (
        f"{'task':<6} {'K':>3} "
        f"{'aIoU':>7} {'aSoft':>7} {'aCent':>7} "
        f"{'oIoU':>7} {'oSoft':>7} {'oCent':>7} {'oPeak':>7}"
    )
    print(header)

    for r in summary_rows:
        print(
            f"{r['task']:<6} {r['K']:>3d} "
            f"{r['agent_iou']:7.3f} "
            f"{r['agent_soft_iou']:7.3f} "
            f"{r['agent_soft_centroid_error']:7.2f} "
            f"{r['interaction_iou']:7.3f} "
            f"{r['interaction_soft_iou']:7.3f} "
            f"{r['interaction_soft_centroid_error']:7.2f} "
            f"{r['interaction_pred_peak']:7.3f}"
        )

    print("\nSaved:", detailed_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()