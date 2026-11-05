import sys
import zlib
import numpy as np

from models.motion_only import parameters as p
sys.modules["parameters"] = p

from data_utils import load_split
from models.motion_only import esn


KS = (10, 30)
OBS_COUNTS = (1, 2, 3, 5)
N = 3000


def sparse_indices(row_id, K, n_obs):
    if n_obs == 1:
        return np.asarray([K - 1], np.int32)

    key = f"{row_id}|K={K}|n={n_obs}"
    seed = zlib.crc32(key.encode()) & 0xffffffff
    rng = np.random.default_rng(seed)

    previous = np.sort(
        rng.choice(
            np.arange(K - 1),
            size=n_obs - 1,
            replace=False,
        )
    )

    return np.concatenate([
        previous.astype(np.int32),
        np.asarray([K - 1], np.int32),
    ])


def evaluate(model, rows, K, n_obs=None):
    correct = []
    joint_rmse = []
    final_rmse = []
    pixel_rmse = []

    for row in rows:
        r = dict(row)

        if n_obs is not None:
            r["_obs_indices"] = sparse_indices(
                row["id"],
                K,
                n_obs,
            )

        # Explicit multimodal availability
        r["_modality_mask"] = np.asarray(
            [1.0, 1.0],
            np.float32,
        )

        pred_img, pred_q, score = esn.predict_trajectory(
            model,
            r,
            K,
            modality_mode="full",
        )

        h = len(pred_q)

        gt_q = np.asarray(
            row["joints"][K:K+h],
            np.float32,
        )

        gt_img = np.asarray(
            row["image"][K:K+h],
            np.float32,
        )

        pred_q = np.asarray(pred_q, np.float32)
        pred_img = np.asarray(pred_img, np.float32)

        correct.append(
            int((score >= 0.5) == int(row["label"]))
        )

        joint_rmse.append(
            np.sqrt(np.mean((pred_q - gt_q) ** 2))
        )

        final_rmse.append(
            np.sqrt(np.mean((pred_q[-1] - gt_q[-1]) ** 2))
        )

        pixel_rmse.append(
            np.sqrt(np.mean((pred_img - gt_img) ** 2))
        )

    return {
        "acc": np.mean(correct),
        "joint": np.mean(joint_rmse),
        "final": np.mean(final_rmse),
        "pixel": np.mean(pixel_rmse),
    }


rows = load_split(
    "test",
    data_path=p.DATA_PATH,
)

model_path = (
    p.MODEL_DIR
    / f"N_{N}"
    / "run_01"
    / "model.npz"
)

print("MODEL:", model_path)
print("ROWS :", len(rows))
print()

model = esn.load_model(model_path)

print("=" * 90)
print("V13 HYBRID CONTEXT TEST | RGB + PROPRIOCEPTION")
print("=" * 90)

for K in KS:
    print()
    print(f"K={K}")
    print("-" * 90)

    base = evaluate(
        model,
        rows,
        K,
        n_obs=None,
    )

    print(
        f"HISTORY | "
        f"acc={base['acc']:.3f} | "
        f"joint={base['joint']:.4f} | "
        f"final={base['final']:.4f} | "
        f"pixel={base['pixel']:.4f}"
    )

    for n in OBS_COUNTS:
        s = evaluate(
            model,
            rows,
            K,
            n_obs=n,
        )

        print(
            f"OBS={n}   | "
            f"acc={s['acc']:.3f} | "
            f"joint={s['joint']:.4f} | "
            f"final={s['final']:.4f} | "
            f"pixel={s['pixel']:.4f}"
        )
