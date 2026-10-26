"""Direct future-frame delta Echo State Network for vision or joint RGB multimodal tasks.

Observes first K frames. RGB is (T,D) float [0,1], joint values are
z-scored using TRAIN ONLY statistics. Returns future RGB and (if enabled)
future normalized joints. Direct readouts by absolute q, no recurrent feedback.
"""
from pathlib import Path
import numpy as np
import cupy as cp
import parameters as p


def configuration():
    mode = p.EXPERIMENT_MODE
    if mode not in ('vision', 'multimodal'):
        raise ValueError('Choose vision or multimodal')
    return mode, bool(p.USE_GOAL)


def input_width():
    mode, goal = configuration()
    return p.IMAGE_DIM + (p.JOINT_DIM if mode == 'multimodal' else 0) + (p.GOAL_DIM if goal else 0)


def _validate(row):
    image = np.asarray(row['image'])
    if image.ndim != 2 or image.shape[1] != p.IMAGE_DIM:
        raise ValueError(f"{row['id']}: expected image (T,{p.IMAGE_DIM}), got {image.shape}")
    if len(image) != len(row['joints']):
        raise ValueError(f"{row['id']}: image/joint frame mismatch")
    if np.asarray(row['goal']).shape != (3,):
        raise ValueError(f"{row['id']}: expected goal XYZ shape (3,)")
    return image


def new_model(N, seed, stats):
    rng = cp.random.RandomState(int(seed))
    dim = input_width()
    Win = rng.uniform(-p.INPUT_SCALE, p.INPUT_SCALE, (N, dim)).astype(cp.float32)
    bias = rng.uniform(-p.INPUT_SCALE, p.INPUT_SCALE, (N,)).astype(cp.float32)
    W = rng.uniform(-1, 1, (N, N)).astype(cp.float32)
    radius = float(cp.max(cp.abs(cp.linalg.eigvals(W))).get())
    W *= cp.float32(p.SPECTRAL_RADIUS / max(radius, 1e-12))
    return dict(Win=Win, W=W, bias=bias, N=N, seed=int(seed),
                mode=p.EXPERIMENT_MODE, use_goal=bool(p.USE_GOAL),
                image_dim=p.IMAGE_DIM, stats=stats, heads={}, accuracy={})


def prepare_observation(row, model, k):
    """One input vector per actual observed image difference (t=1..K-1)."""
    image = cp.asarray(row['image'][:k], cp.float32)
    modalities = [image[1:] - image[:-1]]
    if model['mode'] == 'multimodal':
        raw_q = cp.asarray(row['joints'][:k], cp.float32)
        mean = cp.asarray(model['stats']['qmean'])
        std = cp.asarray(model['stats']['qstd'])
        q = (raw_q - mean) / std
        modalities.append(cp.float32(p.JOINT_INPUT_SCALE) * (q[1:] - q[:-1]))
    if model['use_goal']:
        goal = (cp.asarray(row['goal'], cp.float32) - cp.asarray(model['stats']['gmean'])) / cp.asarray(model['stats']['gstd'])
        modalities.append(cp.broadcast_to(cp.float32(p.GOAL_INPUT_SCALE) * goal, (k-1, 3)))
    return cp.concatenate(modalities, axis=1)


def feature(row, model, k):
    obs = prepare_observation(row, model, k)
    state = cp.zeros((model['N'],), dtype=cp.float32)
    for t in range(len(obs)):
        cand = cp.tanh(model['bias'] + model['Win'] @ obs[t] + model['W'] @ state)
        state = (1 - p.LEAK_RATE) * state + p.LEAK_RATE * cand
    c = p.IMAGE_CHANNELS
    h, w = p.IMAGE_HEIGHT, p.IMAGE_WIDTH
    ah, aw = p.APPEARANCE_HEIGHT, p.APPEARANCE_WIDTH
    img = cp.asarray(row['image'][k - 1], cp.float32).reshape(h, w, c)
    small = img.reshape(ah,h//ah,aw,w//aw,c).mean(axis=(1,3)).ravel() - cp.float32(.5)
    base = [cp.ones((1,), dtype=cp.float32), state, small]
    # Absolute initial robot configuration helps distinguish actions at short K.
    if model['mode'] == 'multimodal':
        q = (cp.asarray(row['joints'][k-1], cp.float32) - cp.asarray(model['stats']['qmean'])) / cp.asarray(model['stats']['qstd'])
        base.append(cp.float32(p.JOINT_INPUT_SCALE)*q)
    if model['use_goal']:
        goal = (cp.asarray(row['goal'], cp.float32) - cp.asarray(model['stats']['gmean'])) / cp.asarray(model['stats']['gstd'])
        base.append(cp.float32(p.GOAL_INPUT_SCALE)*goal)
    return cp.concatenate(base)


def _target(row, k, q, model):
    """Predict cumulative change from the last observed frame, not per-step feedback."""
    rgb_delta = np.asarray(row['image'][q] - row['image'][k-1], np.float32)
    if model['mode'] == 'vision':
        return rgb_delta
    qstd = model['stats']['qstd']
    joint_delta = (row['joints'][q] - row['joints'][k-1]) / qstd
    return np.concatenate((rgb_delta, joint_delta.astype(np.float32)))


def build_model(train, N, training_epsilon=0, augmentation_count=0, seed=42):
    if training_epsilon != 0 or augmentation_count:
        raise NotImplementedError('This first multimodal version deliberately disables augmentation/noise.')
    from data_utils import compute_train_stats
    stats = compute_train_stats(train)
    model = new_model(int(N), int(seed), stats)
    for row in train:
        _validate(row)
    for k in p.READOUT_K:
        rows = [r for r in train if len(r['image']) > k]
        if not rows: raise ValueError(f'No trajectories longer than K={k}')
        X = cp.stack([feature(row, model, k) for row in rows]).astype(cp.float32)
        # A shared classifier: 0=Pick, 1=Push. One ridge regression per K.
        cls_y = cp.asarray(np.array([r['label'] for r in rows], np.float64)[:,None])
        X64 = X.astype(cp.float64)
        gram = X64 @ X64.T + p.READOUT_RIDGE * cp.eye(len(rows), dtype=cp.float64)
        cls_alpha = cp.linalg.solve(gram, cls_y).astype(cp.float32)
        fitted_labels = ((X @ X.T) @ cls_alpha).ravel().get() >= .5
        cls_accuracy = float(np.mean(fitted_labels == np.array([r['label'] for r in rows])))
        max_len = max(len(row['image']) for row in rows)
        alphas, indices = [], []
        for q in range(k, max_len):
            ix = [j for j,row in enumerate(rows) if len(row['image']) > q]
            Xq = X[ix].astype(cp.float64)
            Y = cp.asarray(np.stack([_target(rows[j],k,q,model) for j in ix]), cp.float64)
            A = Xq @ Xq.T + p.READOUT_RIDGE * cp.eye(len(ix), dtype=cp.float64)
            alphas.append(cp.linalg.solve(A, Y).astype(cp.float32))
            indices.append(np.asarray(ix,dtype=np.int32))
        model['heads'][int(k)] = dict(X=X, cls_alpha=cls_alpha,
                                      q_values=np.arange(k,max_len,dtype=np.int32),
                                      alphas=alphas, indices=indices)
        model['accuracy'][int(k)] = cls_accuracy
        print(f'K={k}: n={len(rows)}; final frame index={max_len-1}; training action accuracy={cls_accuracy:.3f}')
    return model


def predict_trajectory(model, row, k, horizon=None):
    """Return (RGB predicted, joints predicted or None, push probability)."""
    head = model['heads'][int(k)]
    max_h = len(head['q_values'])
    horizon = min(len(row['image']) - k, max_h) if horizon is None else int(horizon)
    if horizon < 0 or horizon > max_h: raise ValueError('Invalid horizon')
    x = feature(row, model, k)
    score = float((x @ head['X'].T @ head['cls_alpha']).item())
    base_img = cp.asarray(row['image'][k-1],cp.float32)
    base_q = cp.asarray(row['joints'][k-1],cp.float32)
    outputs_img, outputs_q = [], []
    for step in range(horizon):
        ix = head['indices'][step]
        delta = x @ head['X'][ix].T @ head['alphas'][step]
        outputs_img.append(cp.clip(base_img + cp.clip(delta[:p.IMAGE_DIM], -p.DELTA_CLIP, p.DELTA_CLIP),0,1))
        if model['mode'] == 'multimodal':
            outputs_q.append(base_q + delta[p.IMAGE_DIM:] * cp.asarray(model['stats']['qstd']))
    images = cp.asnumpy(cp.stack(outputs_img)) if outputs_img else np.empty((0,p.IMAGE_DIM),np.float32)
    joints = (cp.asnumpy(cp.stack(outputs_q)) if outputs_q else np.empty((0,p.JOINT_DIM),np.float32)) if model['mode']=='multimodal' else None
    return images, joints, score


def save_model(path, model):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={key:cp.asnumpy(model[key]) for key in ('Win','W','bias')}
    for name, values in model['stats'].items(): payload[f'stats_{name}']=values
    for k,head in model['heads'].items():
        payload[f'X_{k}']=cp.asnumpy(head['X'])
        payload[f'cls_{k}']=cp.asnumpy(head['cls_alpha'])
        payload[f'qvals_{k}']=head['q_values']
        for i,(alpha, ix) in enumerate(zip(head['alphas'],head['indices'])):
            payload[f'a_{k}_{i}']=cp.asnumpy(alpha)
            payload[f'ix_{k}_{i}']=ix
    np.savez_compressed(path, **payload, N=model['N'], seed=model['seed'],
                        mode=model['mode'], use_goal=int(model['use_goal']),
                        image_dim=model['image_dim'], ks=np.asarray(sorted(model['heads']),np.int32),
                        architecture='maniskill_multimodal_direct_v1')


def load_model(path):
    with np.load(path,allow_pickle=False) as d:
        mode=str(d['mode']); use_goal=bool(int(d['use_goal']))
        if mode != p.EXPERIMENT_MODE or use_goal != p.USE_GOAL or int(d['image_dim']) != p.IMAGE_DIM:
            raise ValueError('Model/config mode, goal, or image dimension mismatch')
        stats={name:np.asarray(d[f'stats_{name}'],np.float32) for name in ('qmean','qstd','gmean','gstd')}
        model={name:cp.asarray(d[name],cp.float32) for name in ('Win','W','bias')}
        model.update(N=int(d['N']), seed=int(d['seed']),mode=mode,use_goal=use_goal,
                     image_dim=int(d['image_dim']),stats=stats,heads={})
        for k in d['ks']:
            k=int(k); qvals=np.asarray(d[f'qvals_{k}'],np.int32)
            model['heads'][k] = dict(X=cp.asarray(d[f'X_{k}'],cp.float32),cls_alpha=cp.asarray(d[f'cls_{k}'],cp.float32),
                                    q_values=qvals,
                                    alphas=[cp.asarray(d[f'a_{k}_{i}'],cp.float32) for i in range(len(qvals))],
                                    indices=[np.asarray(d[f'ix_{k}_{i}'],np.int32) for i in range(len(qvals))])
    return model