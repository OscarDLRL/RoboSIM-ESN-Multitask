"""ManiSkill PickCube+PushCube: synchronized original-length RGB, qpos, goals.

No temporal resampling. No data leakage through split normalization.
Import this as data_utils.py when integrating with a multimodal ESN.
"""
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
import parameters as p

RESAMPLE = getattr(Image, 'Resampling', Image).LANCZOS
DEFAULT_PATH = Path(__file__).resolve().parent / 'datasets' / 'maniskill_multimodal' / 'multimodal_original.h5'


def _attr_text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _process_rgb(rgb):
    mode = getattr(p, 'IMAGE_MODE', 'rgb')
    if mode not in ('rgb', 'grayscale'):
        raise ValueError('IMAGE_MODE must be rgb or grayscale')
    h = int(p.IMAGE_HEIGHT)
    w = int(p.IMAGE_WIDTH)
    channels = 3 if mode == 'rgb' else 1
    features = h * w * channels
    frames = np.empty((len(rgb), features), dtype=np.float32)
    for index, frame in enumerate(rgb):
        im = Image.fromarray(frame, 'RGB').convert('RGB' if mode == 'rgb' else 'L')
        im = im.resize((w, h), RESAMPLE)
        frames[index] = np.asarray(im, dtype=np.float32).reshape(-1) / 255.0
    return frames


def _read_group(group, key, split, include_rgb=True):
    source_rgb = group['rgb']
    if source_rgb.ndim != 4 or source_rgb.shape[-1] != 3:
        raise ValueError(f'{key}: invalid RGB shape {source_rgb.shape}')
    joints = np.asarray(group['qpos'][:], dtype=np.float32)
    goal = np.asarray(group['goal_xyz'][:], dtype=np.float32).reshape(3)
    cube = np.asarray(group['cube_xyz'][:], dtype=np.float32)
    length = len(source_rgb)
    if joints.shape != (length, 9) or cube.shape != (length, 3):
        raise ValueError(f'{key}: RGB/joint/cube mismatch')
    if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(goal)):
        raise ValueError(f'{key}: nonfinite features')
    label = int(group.attrs['label'])
    kind = 'pick' if label == 0 else 'push' if label == 1 else None
    if kind is None:
        raise ValueError(f'{key}: invalid label {label}')
    images = _process_rgb(source_rgb[:]) if include_rgb else None
    return {
        'id': key, 'kind': kind, 'label': label, 'split': split,
        'image': images, 'joints': joints, 'goal': goal, 'cube_xyz': cube,
        'length': length,
    }


def load_split(split, data_path=None, include_rgb=True):
    """Load all episodes in split; no resampling and no test-based normalization."""
    if split not in ('train', 'validation', 'test'):
        raise ValueError('split must be train, validation or test')
    path = Path(data_path).expanduser() if data_path is not None else DEFAULT_PATH
    if not path.is_file():
        raise FileNotFoundError(path)
    trajectories = []
    with h5py.File(path, 'r') as handle:
        if 'trajectories' not in handle:
            raise KeyError('Expected /trajectories in ' + str(path))
        root = handle['trajectories']
        for key in sorted(root):
            group = root[key]
            group_split = _attr_text(group.attrs['split'])
            if group_split == split:
                trajectories.append(_read_group(group, key, split, include_rgb))
    if not trajectories:
        raise ValueError(f'No trajectories for split {split} in {path}')
    return trajectories


def load_trajectory(data_path, trajectory_id):
    """Retrieve by new ID, e.g. 'pick_000' or 'push_025'."""
    path = Path(data_path).expanduser()
    name = str(trajectory_id)
    with h5py.File(path, 'r') as handle:
        group = handle['trajectories'][name]
        return _read_group(group, name, _attr_text(group.attrs['split']))


def compute_train_stats(train):
    """Train-only normalization statistics; save these alongside the trained model."""
    joints = np.concatenate([row['joints'] for row in train], axis=0)
    goals = np.stack([row['goal'] for row in train])
    qmean = joints.mean(axis=0).astype(np.float32)
    qstd = np.maximum(joints.std(axis=0), 1e-6).astype(np.float32)
    gmean = goals.mean(axis=0).astype(np.float32)
    gstd = np.maximum(goals.std(axis=0), 1e-6).astype(np.float32)
    return {'qmean': qmean, 'qstd': qstd, 'gmean': gmean, 'gstd': gstd}


def normalize_modalities(data, stats):
    """Return copies with normalized joint/goal data, retaining originals in raw_* fields."""
    result = []
    for row in data:
        item = dict(row)
        item['raw_joints'] = row['joints']
        item['raw_goal'] = row['goal']
        item['joints'] = ((row['joints'] - stats['qmean']) / stats['qstd']).astype(np.float32)
        item['goal'] = ((row['goal'] - stats['gmean']) / stats['gstd']).astype(np.float32)
        result.append(item)
    return result