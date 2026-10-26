#!/usr/bin/env python3
"""Build a paired-task PickCube/PushCube multimodal dataset from official ManiSkill RGB replays.

Keeps original RGB frames and qpos intact. Outputs normalized frame *indices*, not
resampled recordings, so temporal normalization is optional and explicitly offline.
"""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np

TASKS = {'pick': ('PickCube-v1', 'goal_site'), 'push': ('PushCube-v1', 'goal_region')}
RGB_KEY = 'obs/sensor_data/base_camera/rgb'
QPOS_KEY = 'obs/agent/qpos'


def ordered_trajectory_keys(f):
    return sorted((k for k in f if k.startswith('traj_') and k.split('_')[-1].isdigit()),
                  key=lambda k: int(k.split('_')[-1]))


def find_replay(folder):
    files = sorted(folder.glob('trajectory.rgb*.h5'))
    if len(files) != 1:
        raise RuntimeError(f'Expected one RGB replay in {folder}, found {len(files)}: {[p.name for p in files]}')
    return files[0]


def split_ids(count, seed):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(count)
    n_train = int(round(count * 0.8))
    n_val = int(round(count * 0.1))
    return ({int(x): 'train' for x in ids[:n_train]} |
            {int(x): 'validation' for x in ids[n_train:n_train + n_val]} |
            {int(x): 'test' for x in ids[n_train + n_val:]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--count', type=int, default=100, help='Number of trajectories per task (default: 100)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--offline-frames', type=int, default=80, help='Store sampling indices for offline 80-step comparison; no videos are resampled')
    ap.add_argument('--out-dir', type=Path, default=Path('datasets/maniskill_multimodal'))
    ap.add_argument('--demo-root', type=Path, default=Path.home()/'.maniskill/demos')
    args = ap.parse_args()
    if args.count < 1 or args.offline_frames < 2:
        ap.error('--count must be positive and --offline-frames at least 2')
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / 'multimodal_original.h5'
    csv_path = out / 'manifest.csv'
    if h5_path.exists() or csv_path.exists():
        raise FileExistsError('Output already exists; move/delete output files first to prevent accidental overwrite')

    sources = {}
    for label, (task, _) in TASKS.items():
        folder = args.demo_root / task / 'motionplanning'
        original = folder / 'trajectory.h5'
        replay = find_replay(folder)
        if not original.exists():
            raise FileNotFoundError(original)
        with h5py.File(replay) as f:
            keys = ordered_trajectory_keys(f)
            if len(keys) < args.count:
                raise RuntimeError(f'{task} has {len(keys)} RGB trajectories; need {args.count}. Replay at least {args.count} first.')
        sources[label] = (original, replay)
        print(f'{task}: replay={replay.name}')

    rows = []
    try:
        with h5py.File(h5_path, 'w') as target:
            target.attrs.update({'dataset_name': 'ManiSkill PickCube+PushCube multimodal',
                                 'seed': args.seed, 'episodes_per_task': args.count,
                                 'rgb_origin': 'Official ManiSkill use-env-states RGB replays',
                                 'alignment': 'Original native frame indices, no frame resampling',
                                 'offline_frames': args.offline_frames})
            for label, (task, goal_actor) in TASKS.items():
                original_path, replay_path = sources[label]
                assignment = split_ids(args.count, args.seed)
                with h5py.File(original_path, 'r') as original, h5py.File(replay_path, 'r') as replay:
                    keys = ordered_trajectory_keys(replay)[:args.count]
                    for i, key in enumerate(keys):
                        if key not in original:
                            raise KeyError(f'Original trajectory missing: {task}/{key}')
                        observed = replay[key]
                        raw = original[key]
                        rgb = observed[RGB_KEY][:]
                        qpos = observed[QPOS_KEY][:]
                        goal_states = raw[f'env_states/actors/{goal_actor}'][:]
                        # The actor's first 3 state components are world XYZ.
                        goal = goal_states[0, :3].astype(np.float32)
                        cube = raw['env_states/actors/cube'][:, :3].astype(np.float32)
                        steps = len(rgb)
                        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                            raise ValueError(f'{task}/{key}: unexpected RGB {rgb.shape} {rgb.dtype}')
                        if qpos.ndim != 2 or qpos.shape[1] != 9 or len(qpos) != steps:
                            raise ValueError(f'{task}/{key}: unexpected joints {qpos.shape} for RGB {rgb.shape}')
                        if len(goal_states) != steps or len(cube) != steps:
                            raise ValueError(f'{task}/{key}: state/RGB length mismatch')
                        if not np.allclose(goal_states[:, :3], goal, atol=1e-5):
                            raise ValueError(f'{task}/{key}: goal varies within episode')
                        if not np.all(np.isfinite(qpos)):
                            raise ValueError(f'{task}/{key}: qpos contains non-finite values')

                        group_name = f'{label}_{i:03d}'
                        g = target.create_group(f'trajectories/{group_name}')
                        g.attrs['task'] = task
                        g.attrs['label'] = 0 if label == 'pick' else 1
                        g.attrs['split'] = assignment[i]
                        g.attrs['original_key'] = key
                        g.attrs['source_rgb'] = str(replay_path)
                        g.attrs['source_states'] = str(original_path)
                        g.create_dataset('rgb', data=rgb, compression='gzip', compression_opts=1, chunks=(1, *rgb.shape[1:]))
                        g.create_dataset('qpos', data=qpos.astype(np.float32), compression='gzip', compression_opts=1)
                        g.create_dataset('goal_xyz', data=goal)
                        g.create_dataset('cube_xyz', data=cube, compression='gzip', compression_opts=1)
                        # Sampling is retrospective and should NOT be used in real-time K-frame prediction.
                        offline_indices = np.rint(np.linspace(0, steps-1, args.offline_frames)).astype(np.int32)
                        g.create_dataset('offline_normalized_indices', data=offline_indices)
                        success = raw['success'][:].astype(bool)
                        g.attrs['success_ever'] = bool(success.any())
                        g.attrs['success_at_end'] = bool(success[-1]) if len(success) else False
                        rows.append({'id': group_name, 'task': task, 'label': int(g.attrs['label']),
                                     'split': assignment[i], 'original_key': key, 'frames': steps,
                                     'goal_x': float(goal[0]), 'goal_y': float(goal[1]), 'goal_z': float(goal[2]),
                                     'success_ever': int(g.attrs['success_ever']),
                                     'success_at_end': int(g.attrs['success_at_end'])})
                        if (i+1)%10 == 0 or i == 0:
                            print(f'  {task}: {i+1}/{args.count} | latest={key} frames={steps}, split={assignment[i]}')
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        summary = {'total': len(rows), 'per_task': args.count,
                   'splits': {s: sum(r['split'] == s for r in rows) for s in ('train','validation','test')},
                   'native_frame_range': [min(r['frames'] for r in rows), max(r['frames'] for r in rows)],
                   'successful_once': sum(r['success_ever'] for r in rows),
                   'successful_at_end': sum(r['success_at_end'] for r in rows),
                   'notes': ['RGB observations, nine qpos, xyz goal and cube are synchronized',
                             'Original demonstration goal remains variable',
                             'offline_normalized_indices are retrospective; do not use as early-K online observations']}
        (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('\nCOMPLETE:', h5_path, '\nMANIFEST:', csv_path, '\n', json.dumps(summary, indent=2))
    except Exception:
        # Never leave an apparently complete dataset after a partial failure.
        h5_path.unlink(missing_ok=True)
        raise

if __name__ == '__main__':
    main()