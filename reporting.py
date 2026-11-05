"""Summaries, plots and qualitative GIFs for any ESN model."""

import argparse, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import metrics
from model_loader import load_model_package

METRICS = (
    "correct","pixel_rmse","delta_image_rmse","moving_region_rmse",
    "motion_f1","joint_mae","joint_rmse"
)


def frame_image(frame, p):
    a = np.asarray(frame, np.float32)
    h, w, c = p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.IMAGE_CHANNELS
    if a.size != h*w*c:
        raise ValueError("Invalid frame size")
    a = np.clip(a.reshape(h,w,c), 0, 1)
    if c == 1:
        a = a[...,0]
    return Image.fromarray(np.uint8(np.rint(a*255))).resize((384,384), Image.Resampling.NEAREST)


def show_example(file, out, p):
    with np.load(file, allow_pickle=False) as z:
        gt, pred, prev = z["true_image"], z["pred_image"], z["previous_image"]
        gq = z["true_joints"] if "true_joints" in z else np.empty((0,9))
        pq = z["pred_joints"] if "pred_joints" in z else np.empty((0,9))

    if not len(gt) or gt.shape != pred.shape:
        return False

    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (g, y) in enumerate(zip(gt, pred), 1):
        panel = Image.new("RGB", (774,414), "white")
        panel.paste(frame_image(g,p), (0,30))
        panel.paste(frame_image(y,p), (390,30))
        d = ImageDraw.Draw(panel)
        d.text((8,9), f"GROUND TRUTH | +{i}", fill="black")
        d.text((399,9), f"ESN | {file.stem}", fill="black")
        frames.append(panel)

    frames[0].save(
        out/f"{file.stem}.gif", save_all=True, append_images=frames[1:],
        duration=getattr(p,"GIF_FRAME_MS",120), loop=0
    )

    pixel, delta = metrics.per_frame_metrics(gt, pred, prev, p.IMAGE_HEIGHT, p.IMAGE_WIDTH)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(np.arange(1,len(pixel)+1), pixel, label="Pixel RMSE")
    ax.plot(np.arange(1,len(delta)+1), delta, label="Delta Image RMSE")
    ax.set(xlabel="Forecast step", ylabel="RMSE [0,1]", title=file.stem)
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(out/f"{file.stem}_horizon.png", dpi=150); plt.close(fig)

    if len(gq) and gq.shape == pq.shape:
        fig, axes = plt.subplots(3,3, figsize=(11,8), sharex=True)
        for j, ax in enumerate(axes.flat):
            if j >= gq.shape[1]:
                ax.axis("off"); continue
            ax.plot(gq[:,j], label="Ground truth")
            ax.plot(pq[:,j], "--", label="ESN")
            ax.set_title(f"Joint {j+1}"); ax.grid(alpha=.2)
        axes.flat[0].legend(); fig.tight_layout()
        fig.savefig(out/f"{file.stem}_joints.png", dpi=150); plt.close(fig)

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("baseline","motion_only","timeindexed"), default="baseline")
    ap.add_argument("--split", choices=("validation","test"), default="validation")
    ap.add_argument("--neurons", type=int)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--max-gifs-per-k-task", type=int, default=2)
    a = ap.parse_args()

    p, esn = load_model_package(a.model)
    sys.modules["parameters"], sys.modules["esn"] = p, esn

    n = a.neurons or p.NEURON_COUNTS[0]
    root = p.TEST_RESULTS_DIR / a.split / f"N_{n}" / f"run_{a.run:02d}"
    csv_path = root / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    data = pd.read_csv(csv_path)
    needed = {"task","K","correct","pixel_rmse","delta_image_rmse"}
    if not needed.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(needed-set(data.columns))}")

    out = root / "reporting"
    out.mkdir(parents=True, exist_ok=True)
    cols = [x for x in METRICS if x in data.columns]

    by_task = data.groupby(["task","K"], as_index=False)[cols].mean(numeric_only=True)
    by_k = data.groupby("K", as_index=False)[cols].mean(numeric_only=True)
    by_task.rename(columns={"correct":"action_accuracy"}, inplace=True)
    by_k.rename(columns={"correct":"action_accuracy"}, inplace=True)

    by_task.to_csv(out/"summary_by_task_k.csv", index=False)
    by_k.to_csv(out/"summary_by_k.csv", index=False)

    print(f"\nMODEL={a.model}\nARCH={esn.ARCHITECTURE}\n")
    print("SUMMARY BY TASK AND K:")
    print(by_task.round(4).to_string(index=False))

    for col in ("pixel_rmse","delta_image_rmse","moving_region_rmse","joint_mae","joint_rmse","action_accuracy"):
        if col not in by_task.columns:
            continue
        fig, ax = plt.subplots(figsize=(7,4))
        for task, d in by_task.groupby("task"):
            ax.plot(d["K"], d[col], "-o", label=task)
        ax.set(xlabel="K observed frames", ylabel=col, title=f"{col} by K")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(out/f"{col}_by_k.png", dpi=150); plt.close(fig)

    counts = {}
    for file in sorted(root.glob("K_*/predictions/*.npz")):
        key = (file.parent.parent.name, file.stem.split("_")[0])
        if counts.get(key,0) >= a.max_gifs_per_k_task:
            continue
        if show_example(file, out/key[0]/key[1], p):
            counts[key] = counts.get(key,0) + 1

    print("\nSaved:", out)
    print("Physical task success requires sim_rollout.py.")


if __name__ == "__main__":
    main()
