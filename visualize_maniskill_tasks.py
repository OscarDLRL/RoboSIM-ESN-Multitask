#!/usr/bin/env python3
"""Visualize actual RGB trajectories in datasets/maniskill_multimodal/multimodal_original.h5.

Examples:
  python visualize_maniskill_multimodal.py
  python visualize_maniskill_multimodal.py --split validation
  python visualize_maniskill_multimodal.py --pick pick_005 --push push_005
  python visualize_maniskill_multimodal.py --list --split test
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
from PIL import Image, ImageDraw

TASKS = ("pick", "push")


def list_ids(h5, label, split):
    root = h5["trajectories"]
    result = []
    for key in sorted(root.keys()):
        if not key.startswith(label + "_"):
            continue
        if split is None or root[key].attrs.get("split") == split:
            result.append(key)
    return result


def choose(h5, label, requested, split):
    eligible = list_ids(h5, label, split)
    if not eligible:
        raise ValueError(f"No {label} demonstrations in split={split}")
    if requested is None:
        return eligible[0]
    key = f"{label}_{int(requested):03d}" if requested.isdigit() else requested
    if key not in eligible:
        raise ValueError(f"{key} is not in split={split}; choose from: {', '.join(eligible)}")
    return key


def read_group(h5, key):
    g = h5[f"trajectories/{key}"]
    rgb = g["rgb"][:]
    goal = g["goal_xyz"][:]
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"{key}: unsupported rgb shape {rgb.shape}")
    if len(rgb) != len(g["qpos"]):
        raise ValueError(f"{key}: RGB/qpos mismatch")
    return {"id": key, "frames": rgb, "goal": goal, "split": str(g.attrs["split"])}


def frame_panel(series, t, width):
    frames = series["frames"]
    index = min(t, len(frames) - 1)
    im = Image.fromarray(frames[index].astype(np.uint8), "RGB")
    im = im.resize((width, width), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (width, width + 66), "white")
    panel.paste(im, (0, 66))
    d = ImageDraw.Draw(panel)
    d.text((9, 7), f'{series["id"]} / {series["split"]}', fill="black")
    d.text((9, 26), f'Frame {index+1}/{len(frames)}' + (' (finished)' if t >= len(frames) else ''), fill="black")
    d.text((9, 45), 'Goal XYZ: ' + ', '.join(f'{float(x):+.3f}' for x in series["goal"]), fill="black")
    return panel


def save_gif(frames, path, frame_ms):
    if not frames:
        raise ValueError('No frames for GIF')
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0,
                   optimize=False)
    print('Saved:', path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, default=Path.cwd())
    ap.add_argument('--dataset', type=Path, default=None)
    ap.add_argument('--split', choices=('train', 'validation', 'test'), default=None,
                    help='If supplied, select and validate trajectories from only this split')
    ap.add_argument('--pick', default=None, help='pick_005 or 5; default first eligible')
    ap.add_argument('--push', default=None, help='push_005 or 5; default first eligible')
    ap.add_argument('--list', action='store_true', help='List available trajectory IDs and exit')
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--frame-ms', type=int, default=100)
    args = ap.parse_args()
    if args.size < 64 or args.frame_ms < 10:
        ap.error('--size must be >=64 and --frame-ms >=10')
    project = args.project.expanduser().resolve()
    path = args.dataset.expanduser().resolve() if args.dataset else project/'datasets/maniskill_multimodal/multimodal_original.h5'
    if not path.exists():
        ap.error(f'Dataset not found: {path}')
    with h5py.File(path, 'r') as h5:
        if args.list:
            for label in TASKS:
                print(label + ':', ', '.join(list_ids(h5, label, args.split)))
            return
        selected = {label: choose(h5, label, getattr(args, label), args.split) for label in TASKS}
        series = {label: read_group(h5, selected[label]) for label in TASKS}
    out = project/'datasets/visualizations'
    for label, seq in series.items():
        print(f'{label}: {seq["id"]}, frames={len(seq["frames"])}, RGB={seq["frames"].shape[1:]}, '
              f'goal={np.round(seq["goal"],3)}, split={seq["split"]}')
        save_gif([frame_panel(seq, t, args.size) for t in range(len(seq['frames']))],
                 out/f'{seq["id"]}.gif', args.frame_ms)
    n = max(len(s['frames']) for s in series.values())
    combined = []
    for t in range(n):
        p = frame_panel(series['pick'], t, args.size)
        q = frame_panel(series['push'], t, args.size)
        side = Image.new('RGB', (args.size * 2 + 6, args.size + 66), 'white')
        side.paste(p, (0, 0))
        side.paste(q, (args.size+6, 0))
        combined.append(side)
    save_gif(combined, out/'Pick_vs_Push_multimodal.gif', args.frame_ms)
    print('Note: each original frame is preserved; the shorter series holds its last frame.')


if __name__ == '__main__':
    main()