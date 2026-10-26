"""Evaluate ESN forecasts on all trajectories in selected holdout split.

No future cube positions or simulator success are inferred from RGB forecasts.
"""
import argparse
import csv
import numpy as np
import cupy as cp
import parameters as p
import metrics
from data_utils import load_split
from esn import load_model, predict_trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('validation', 'test'), default='validation')
    ap.add_argument('--neurons', type=int, default=p.NEURON_COUNTS[0])
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--save-predictions', type=int, default=2)
    args = ap.parse_args()
    data = load_split(args.split, data_path=p.DATA_PATH)
    model_path = p.MODEL_DIR / f'N_{args.neurons}' / f'run_{args.run:02d}' / 'model.npz'
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = load_model(model_path)
    out = p.TEST_RESULTS_DIR / args.split / f'N_{args.neurons}' / f'run_{args.run:02d}'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for k in p.TEST_K:
        saved = {'pick': 0, 'push': 0}
        for row in data:
            if row['length'] <= k:
                continue
            pred_img, pred_q, prob = predict_trajectory(model, row, k)
            gt = row['image'][k:k+len(pred_img)]
            if not len(gt) or len(gt) != len(pred_img):
                continue
            prev = row['image'][k-1]
            visual = metrics.prediction_metrics(gt, pred_img, prev, p.IMAGE_HEIGHT,
                                                p.IMAGE_WIDTH, getattr(p, 'MOTION_THRESHOLD', .08))
            rec = dict(split=args.split, task=row['kind'], trajectory=row['id'], K=k,
                       frames=len(pred_img), push_probability=float(prob),
                       true_label=int(row['label']), predicted_label=int(prob >= .5),
                       correct=int(int(prob >= .5) == int(row['label'])), **visual)
            if pred_q is not None:
                qtrue = np.asarray(row['joints'][k:k+len(pred_q)], np.float32)
                qpred = np.asarray(pred_q, np.float32)
                if qtrue.shape != qpred.shape:
                    raise ValueError(f"{row['id']} joints shape mismatch {qtrue.shape} / {qpred.shape}")
                rec['joint_mae'] = float(np.mean(np.abs(qpred-qtrue)))
                rec['joint_rmse'] = float(np.sqrt(np.mean((qpred-qtrue)**2)))
            else:
                qtrue = np.empty((0, 9), np.float32)
                qpred = np.empty((0, 9), np.float32)
                rec.update(joint_mae=np.nan, joint_rmse=np.nan)
            records.append(rec)
            if saved[row['kind']] < args.save_predictions:
                target = out / f'K_{k}' / 'predictions'
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / f"{row['id']}.npz",
                                    previous_image=prev, true_image=gt, pred_image=pred_img,
                                    true_joints=qtrue, pred_joints=qpred,
                                    goal=row['goal'], action=row['kind'], K=k)
                saved[row['kind']] += 1
        subset = [r for r in records if r['K'] == k]
        if subset:
            print(f"K={k}: n={len(subset)} accuracy={np.mean([r['correct'] for r in subset]):.3f} "
                  f"pixel={np.mean([r['pixel_rmse'] for r in subset]):.4f} "
                  f"delta={np.mean([r['delta_image_rmse'] for r in subset]):.4f}")
    if not records:
        raise ValueError('No predictions generated')
    target = out / 'metrics.csv'
    with target.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('Saved:', target)
    del model
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()