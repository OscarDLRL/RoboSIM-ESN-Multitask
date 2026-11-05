import sys
import numpy as np

from models.motion_only import parameters as p
sys.modules["parameters"] = p

from models.motion_only import esn
from data_utils import load_split

from query_v14 import _v13_context
from query_v14c import (
    load_query_head,
    predict_query_from_context,
)

N = 3000
RUN = 1

CONTEXT_FRACTIONS = {
    1: (0.50,),
    3: (0.10, 0.40, 0.70),
    5: (0.05, 0.25, 0.45, 0.65, 0.85),
}


def make_obs_indices(T, fractions):
    idx = np.asarray(
        [
            int(round(f * (T - 1)))
            for f in fractions
        ],
        dtype=np.int32,
    )

    return np.unique(
        np.clip(idx, 0, T - 1)
    )


def rmse(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)

    return float(
        np.sqrt(
            np.mean(
                (a - b) ** 2
            )
        )
    )


def region_for_target(target, obs_idx):
    if target < int(obs_idx[0]):
        return "before"

    if target > int(obs_idx[-1]):
        return "after"

    if np.any(obs_idx == target):
        return "observed"

    return "between"


model_path = (
    p.MODEL_DIR
    / f"N_{N}"
    / f"run_{RUN:02d}"
    / "model.npz"
)

query_path = (
    p.RESULTS_DIR
    / "query_v14c_1000"
    / f"N_{N}"
    / f"run_{RUN:02d}"
    / "query_head.npz"
)

print("V13 MODEL :", model_path)
print("V14C HEAD :", query_path)

model = esn.load_model(
    model_path
)

head = load_query_head(
    query_path
)

rows = load_split(
    "test",
    data_path=p.DATA_PATH,
)

records = []

for n_obs, fractions in CONTEXT_FRACTIONS.items():

    for row in rows:

        T = int(
            len(row["image"])
        )

        obs_idx = make_obs_indices(
            T,
            fractions,
        )

        # Context computed once
        context = _v13_context(
            row,
            model,
            obs_idx,
        )

        for target in range(T):

            pred_img, pred_q = (
                predict_query_from_context(
                    row,
                    model,
                    head,
                    context,
                    obs_idx,
                    target,
                )
            )

            gt_img = np.asarray(
                row["image"][target],
                np.float32,
            )

            gt_q = np.asarray(
                row["joints"][target],
                np.float32,
            )

            # Nearest-observation baseline
            nearest = int(
                obs_idx[
                    np.argmin(
                        np.abs(
                            obs_idx - target
                        )
                    )
                ]
            )

            base_img = np.asarray(
                row["image"][nearest],
                np.float32,
            )

            base_q = np.asarray(
                row["joints"][nearest],
                np.float32,
            )

            q_error = rmse(
                pred_q,
                gt_q,
            )

            pixel_error = rmse(
                pred_img,
                gt_img,
            )

            base_q_error = rmse(
                base_q,
                gt_q,
            )

            base_pixel_error = rmse(
                base_img,
                gt_img,
            )

            region = region_for_target(
                target,
                obs_idx,
            )

            records.append(
                {
                    "n_obs": n_obs,
                    "region": region,

                    "q": q_error,
                    "pixel": pixel_error,

                    "base_q": base_q_error,
                    "base_pixel": base_pixel_error,

                    "win_q":
                        q_error
                        < base_q_error,

                    "win_pixel":
                        pixel_error
                        < base_pixel_error,
                }
            )


def report(name, rr):
    if not rr:
        return

    print(
        f"{name:11s} | "
        f"q={np.mean([x['q'] for x in rr]):.4f} | "
        f"pixel={np.mean([x['pixel'] for x in rr]):.4f} | "
        f"near_q={np.mean([x['base_q'] for x in rr]):.4f} | "
        f"near_pixel={np.mean([x['base_pixel'] for x in rr]):.4f} | "
        f"win_q={100*np.mean([x['win_q'] for x in rr]):5.1f}% | "
        f"win_pixel={100*np.mean([x['win_pixel'] for x in rr]):5.1f}%"
    )


print()
print("=" * 110)
print("V14c FULL TRAJECTORY RECONSTRUCTION | TEST")
print("baseline = nearest observed state")
print("=" * 110)

for n_obs in CONTEXT_FRACTIONS:

    rr = [
        x for x in records
        if x["n_obs"] == n_obs
    ]

    unobserved = [
        x for x in rr
        if x["region"] != "observed"
    ]

    print()
    print(
        f"OBS={n_obs} | "
        f"context fractions="
        f"{CONTEXT_FRACTIONS[n_obs]}"
    )

    report(
        "ALL_UNOBS",
        unobserved,
    )

    for region in (
        "before",
        "between",
        "after",
        "observed",
    ):

        sub = [
            x for x in rr
            if x["region"] == region
        ]

        report(
            region.upper(),
            sub,
        )
