import sys

import numpy as np
import cupy as cp


from models.motion_only import parameters as p

p.RIDGE_CG_MAX_ITER = 1000

sys.modules["parameters"] = p


from models.motion_only import esn
from data_utils import load_split

from query_v14 import (
    context_indices,
    query_targets,
    _v13_context,
)

from query_v14c import (
    residual_query_raw_from_context,
    nearest_anchor,
    query_block_sizes,
    save_query_head,
)


N = 3000
RUN = 1

OBS_COUNTS = (
    1,
    2,
    3,
    4,
    5,
)


def main():

    # ========================================================
    # Frozen V13
    # ========================================================

    model_path = (
        p.MODEL_DIR
        / f"N_{N}"
        / f"run_{RUN:02d}"
        / "model.npz"
    )

    print(
        "V13 MODEL:",
        model_path,
    )

    model = esn.load_model(
        model_path
    )

    # ========================================================
    # Training trajectories
    # ========================================================

    rows = []

    for split in p.TRAIN_SPLITS:

        rows.extend(
            load_split(
                split,
                data_path=p.DATA_PATH,
            )
        )

    print(
        "TRAJECTORIES:",
        len(rows),
    )

    basis = model[
        "bases"
    ]["all"]

    qmean = np.asarray(
        model["stats"]["qmean"],
        np.float32,
    )

    qstd = np.asarray(
        model["stats"]["qstd"],
        np.float32,
    )

    # ========================================================
    # Query training dataset
    # ========================================================

    Xraw = []
    Yq = []
    Yrgb = []
    groups = []

    for row in rows:

        T = int(
            len(row["image"])
        )

        for n_obs in OBS_COUNTS:

            obs_idx = context_indices(
                row["id"],
                T,
                n_obs,
            )

            # Reservoir/context computed ONCE
            # for all targets belonging to this sparse context.
            context = _v13_context(
                row,
                model,
                obs_idx,
            )

            targets = query_targets(
                row["id"],
                obs_idx,
                T,
            )

            for target in targets:

                target = int(
                    target
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

                Xraw.append(
                    cp.asnumpy(raw)
                )

                # --------------------------------------------
                # JOINT TARGET = delta from nearest anchor
                # normalized with V13 qstd
                # --------------------------------------------

                q_target = np.asarray(
                    row["joints"][target],
                    np.float32,
                )

                q_anchor = np.asarray(
                    row["joints"][anchor],
                    np.float32,
                )

                dq = (
                    q_target
                    - q_anchor
                ) / qstd

                Yq.append(
                    dq.astype(
                        np.float32
                    )
                )

                # --------------------------------------------
                # RGB TARGET = delta image in PCA space
                # --------------------------------------------

                image_target = cp.asarray(
                    row["image"][target],
                    cp.float32,
                )

                image_anchor = cp.asarray(
                    row["image"][anchor],
                    cp.float32,
                )

                dimage = (
                    image_target
                    - image_anchor
                )

                dcoef = (
                    dimage
                    @ basis
                )

                Yrgb.append(
                    cp.asnumpy(
                        dcoef
                    ).astype(
                        np.float32
                    )
                )

                # grouped CV:
                # queries from same original trajectory
                # stay together.
                groups.append(
                    str(row["id"])
                )

    raw = cp.asarray(
        np.stack(Xraw),
        cp.float32,
    )

    Yq = cp.asarray(
        np.stack(Yq),
        cp.float32,
    )

    Yrgb = cp.asarray(
        np.stack(Yrgb),
        cp.float32,
    )

    groups = np.asarray(
        groups
    )

    print(
        "QUERY ROWS:",
        len(groups),
    )

    print(
        "RAW SHAPE:",
        raw.shape,
    )

    print(
        "Q TARGET SHAPE:",
        Yq.shape,
    )

    print(
        "RGB TARGET SHAPE:",
        Yrgb.shape,
    )

    # ========================================================
    # Feature normalization
    # ========================================================

    sizes = query_block_sizes(
        model
    )

    weights = dict(
        esn._block_weights()
    )

    weights["time"] = 1.0

    grid = esn._ridge_grid()

    seed = int(
        model["seed"]
    )

    fmean, fscale = (
        esn._feature_normalizer(
            raw,
            sizes,
            weights,
        )
    )

    X = esn._apply_normalizer(
        raw,
        fmean,
        fscale,
    ).astype(
        cp.float32
    )

    # ========================================================
    # Delta-q Ridge
    # ========================================================

    print()
    print(
        "Selecting Ridge for delta-q..."
    )

    rel_q, cv_q = (
        esn._select_ridge_relative_lambda(
            raw,
            Yq,
            groups,
            sizes,
            weights,
            grid,
            seed + 3401,
        )
    )

    (
        q_W,
        q_b,
        lam_q,
        solver_q,
        it_q,
        res_q,
    ) = esn._ridge_matrixfree_fit(
        X,
        Yq,
        rel_q,
    )

    # ========================================================
    # Delta-RGB Ridge
    # ========================================================

    print(
        "Selecting Ridge for delta-RGB..."
    )

    rel_rgb, cv_rgb = (
        esn._select_ridge_relative_lambda(
            raw,
            Yrgb,
            groups,
            sizes,
            weights,
            grid,
            seed + 3409,
        )
    )

    (
        rgb_W,
        rgb_b,
        lam_rgb,
        solver_rgb,
        it_rgb,
        res_rgb,
    ) = esn._ridge_matrixfree_fit(
        X,
        Yrgb,
        rel_rgb,
    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print(
        f"delta-q CV RMSE       : "
        f"{cv_q:.6f}"
    )

    print(
        f"delta-RGB PCA CV RMSE : "
        f"{cv_rgb:.6f}"
    )

    print(
        f"q Ridge rel           : "
        f"{rel_q:.3e}"
    )

    print(
        f"RGB Ridge rel         : "
        f"{rel_rgb:.3e}"
    )

    print(
        f"q PCG                 : "
        f"{it_q} iterations, "
        f"residual={res_q:.3e}"
    )

    print(
        f"RGB PCG               : "
        f"{it_rgb} iterations, "
        f"residual={res_rgb:.3e}"
    )

    # ========================================================
    # Save V14c only
    # ========================================================

    output_path = (
        p.RESULTS_DIR
        / "query_v14c_1000"
        / f"N_{N}"
        / f"run_{RUN:02d}"
        / "query_head.npz"
    )

    save_query_head(
        output_path,
        {
            "fmean":
                cp.asnumpy(fmean),

            "fscale":
                cp.asnumpy(fscale),

            "q_W":
                cp.asnumpy(q_W),

            "q_b":
                cp.asnumpy(q_b),

            "rgb_W":
                cp.asnumpy(rgb_W),

            "rgb_b":
                cp.asnumpy(rgb_b),

            "basis":
                cp.asnumpy(basis),

            "qmean":
                qmean,

            "qstd":
                qstd,
        },
    )

    print()
    print(
        "SAVED QUERY HEAD:",
        output_path,
    )


if __name__ == "__main__":
    main()
