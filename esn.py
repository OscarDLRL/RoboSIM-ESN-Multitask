"""Direct future-frame delta ESN (v5: PCA targets + class-conditional memory + interaction context).

Observes first K frames. Escena global (RGB) + contexto de interaccion derivado de
`interaction_mask`: [centroid_x, centroid_y, area] por frame, SOLO de frames observados.
El contexto entra al reservoir (frames 1..K-1) y al readout (frame K-1).
Sin coordenadas privilegiadas del simulador.
"""
from pathlib import Path
import numpy as np
import cupy as cp
import parameters as p

ARCHITECTURE = 'maniskill_multimodal_direct_pca_classcond_interaction_v5'
HEAD_ARRAYS = ('X', 'fmean', 'fscale', 'rgb_alpha', 'rgb_mean')
CONTEXT_DIM = 3


# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
def configuration():
    mode = p.EXPERIMENT_MODE
    if mode not in ('vision', 'multimodal'):
        raise ValueError('Choose vision or multimodal')
    return mode, bool(p.USE_GOAL)


def _use_interaction():
    return bool(getattr(p, 'USE_INTERACTION_CONTEXT', True))


def input_width():
    mode, goal = configuration()
    return (
        p.IMAGE_DIM
        + (p.JOINT_DIM if mode == 'multimodal' else 0)
        + (CONTEXT_DIM if _use_interaction() else 0)
        + (p.GOAL_DIM if goal else 0)
    )


def _block_weights():
    default = {'state': 1.0, 'small': 1.0, 'q': 2.0, 'interaction': 2.0, 'goal': 2.0}
    default.update(getattr(p, 'BLOCK_WEIGHTS', {}))
    return default


def _ridge_grid():
    return np.asarray(getattr(p, 'RIDGE_GRID', np.logspace(-10, 1, 12)), np.float64)


def _class_conditional():
    return bool(getattr(p, 'CLASS_CONDITIONAL', True))


# ----------------------------------------------------------------------------
# Contexto de interaccion (desde la mascara)
# ----------------------------------------------------------------------------
def _interaction_context(row):
    """(T,3): [centroid_x, centroid_y, area_fraction] desde máscara nativa."""

    key = getattr(
        p,
        "INTERACTION_MASK_KEY",
        "interaction_mask"
    )

    if key not in row:
        raise KeyError(
            f"{row.get('id', '?')}: no tiene '{key}'"
        )

    m = np.asarray(
        row[key],
        np.float32
    )

    if m.ndim == 4 and m.shape[-1] == 1:
        m = m[..., 0]

    if m.ndim != 3:
        raise ValueError(
            f"{row.get('id', '?')}: "
            f"{key} debe ser (T,H,W), got {m.shape}"
        )

    T = len(row["image"])

    if len(m) != T:
        raise ValueError(
            f"{row.get('id', '?')}: "
            f"mask/image mismatch {len(m)} vs {T}"
        )

    # Usamos la resolución NATIVA de la máscara.
    h, w = m.shape[1:]

    m = np.clip(
        m,
        0.0,
        1.0
    )

    area_px = m.sum(
        axis=(1, 2)
    )

    xs = np.linspace(
        -1.0,
        1.0,
        w,
        dtype=np.float32
    )[None, None, :]

    ys = np.linspace(
        -1.0,
        1.0,
        h,
        dtype=np.float32
    )[None, :, None]

    denom = np.maximum(
        area_px,
        1e-6
    )

    cx = (
        (m * xs).sum(axis=(1, 2))
        / denom
    )

    cy = (
        (m * ys).sum(axis=(1, 2))
        / denom
    )

    # Si temporalmente no se ve el objeto,
    # conservar el último centroide válido.
    last_cx = 0.0
    last_cy = 0.0

    for t in range(T):
        if area_px[t] < 1e-6:
            cx[t] = last_cx
            cy[t] = last_cy
        else:
            last_cx = float(cx[t])
            last_cy = float(cy[t])

    area_fraction = (
        area_px
        / float(h * w)
    )

    ctx = np.stack(
        [
            cx,
            cy,
            area_fraction
        ],
        axis=1
    ).astype(np.float32)

    if not np.isfinite(ctx).all():
        raise ValueError(
            f"{row.get('id', '?')}: "
            "interaction context no finito"
        )

    return ctx

def _norm_context(row, model, k):
    """(k,3) cupy: contexto de los primeros k frames, estandarizado con stats del train."""
    ctx = cp.asarray(_interaction_context(row)[:k], cp.float32)
    return (ctx - cp.asarray(model['stats']['imean'])) / cp.asarray(model['stats']['istd'])


def _validate(row):
    image = np.asarray(row['image'])
    if image.ndim != 2 or image.shape[1] != p.IMAGE_DIM:
        raise ValueError(f"{row['id']}: expected image (T,{p.IMAGE_DIM}), got {image.shape}")
    if len(image) != len(row['joints']):
        raise ValueError(f"{row['id']}: image/joint frame mismatch")
    if np.asarray(row['goal']).shape != (3,):
        raise ValueError(f"{row['id']}: expected goal XYZ shape (3,)")
    if _use_interaction():
        _interaction_context(row)
    return image


# ----------------------------------------------------------------------------
# ESN
# ----------------------------------------------------------------------------
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
                use_interaction=_use_interaction(),
                image_dim=p.IMAGE_DIM, stats=stats, heads={}, accuracy={}, bases={})


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
    if model['use_interaction']:
        scale = cp.float32(getattr(p, 'INTERACTION_INPUT_SCALE', 0.5))
        modalities.append(scale * _norm_context(row, model, k)[1:])
    if model['use_goal']:
        goal = (cp.asarray(row['goal'], cp.float32) - cp.asarray(model['stats']['gmean'])) / cp.asarray(model['stats']['gstd'])
        modalities.append(cp.broadcast_to(cp.float32(p.GOAL_INPUT_SCALE) * goal, (k - 1, 3)))
    return cp.concatenate(modalities, axis=1)


def block_sizes(model):
    """Lista ordenada (nombre, tamano) de los bloques del feature."""
    sizes = [('state', int(model['N'])),
             ('small', p.APPEARANCE_HEIGHT * p.APPEARANCE_WIDTH * p.IMAGE_CHANNELS)]
    if model['mode'] == 'multimodal':
        sizes.append(('q', int(p.JOINT_DIM)))
    if model['use_interaction']:
        sizes.append(('interaction', CONTEXT_DIM))
    if model['use_goal']:
        sizes.append(('goal', 3))
    return sizes


def raw_feature(row, model, k):
    """Feature SIN centrar ni escalar: [state, small, (q), (interaction), (goal)]."""
    obs = prepare_observation(row, model, k)
    state = cp.zeros((model['N'],), dtype=cp.float32)
    for t in range(len(obs)):
        cand = cp.tanh(model['bias'] + model['Win'] @ obs[t] + model['W'] @ state)
        state = (1 - p.LEAK_RATE) * state + p.LEAK_RATE * cand
    c = p.IMAGE_CHANNELS
    h, w = p.IMAGE_HEIGHT, p.IMAGE_WIDTH
    ah, aw = p.APPEARANCE_HEIGHT, p.APPEARANCE_WIDTH
    img = cp.asarray(row['image'][k - 1], cp.float32).reshape(h, w, c)
    small = img.reshape(ah, h // ah, aw, w // aw, c).mean(axis=(1, 3)).ravel() - cp.float32(.5)
    parts = [state, small]
    if model['mode'] == 'multimodal':
        q = (cp.asarray(row['joints'][k - 1], cp.float32) - cp.asarray(model['stats']['qmean'])) / cp.asarray(model['stats']['qstd'])
        parts.append(q)
    if model['use_interaction']:
        parts.append(_norm_context(row, model, k)[k - 1])
    if model['use_goal']:
        goal = (cp.asarray(row['goal'], cp.float32) - cp.asarray(model['stats']['gmean'])) / cp.asarray(model['stats']['gstd'])
        parts.append(goal)
    return cp.concatenate(parts)


def _norm(raw, head):
    """Centra y balancea un feature crudo con las stats de una cabeza."""
    return (raw - head['fmean']) * head['fscale']


def feature(row, model, k, head=None):
    head = model['heads'][int(k)] if head is None else head
    return _norm(raw_feature(row, model, k), head)


# ----------------------------------------------------------------------------
# Base PCA de la imagen
# ----------------------------------------------------------------------------
def _image_basis(rows, rank, seed):
    """PCA (IMAGE_DIM, r) de los deltas image[q]-image[k-1] reales de `rows`."""
    rng = np.random.RandomState(int(seed))
    imgs = [np.asarray(row['image'], np.float32) for row in rows]
    pairs = []
    for i, im in enumerate(imgs):
        T = len(im)
        for k in p.READOUT_K:
            if T > k:
                pairs.extend((i, k, q) for q in range(k, T))
    max_samples = int(getattr(p, 'IMAGE_BASIS_SAMPLES', 4000))
    if len(pairs) > max_samples:
        sel = rng.choice(len(pairs), max_samples, replace=False)
        pairs = [pairs[j] for j in sel]
    D = cp.asarray(np.stack([imgs[i][q] - imgs[i][k - 1] for i, k, q in pairs]).astype(np.float32))
    _, _, Vt = cp.linalg.svd(D, full_matrices=False)
    rank = int(min(rank, Vt.shape[0]))
    return cp.ascontiguousarray(Vt[:rank].T).astype(cp.float32)  # (IMAGE_DIM, r)


# ----------------------------------------------------------------------------
# Ridge dual con LOO
# ----------------------------------------------------------------------------
def _ridge_loo(U, s, Y, lam_abs):
    """Elige lambda por LOO. Gram G = U diag(s) U^T ya incluye el intercepto.

    Residuo LOO exacto: e_i = alpha_i / (G + lam I)^-1_ii (estable cuando lam -> 0).
    Devuelve (alpha, lambda, loo_residuals) con el mejor lambda.
    """
    UtY = U.T @ Y
    U2 = U * U
    best = None
    for lam in lam_abs:
        inv = 1.0 / (s + lam)
        alpha = U @ (inv[:, None] * UtY)
        resid = alpha / (U2 @ inv)[:, None]
        err = float(cp.mean(resid ** 2))
        if best is None or err < best[0]:
            best = (err, float(lam), alpha, resid)
    _, lam, alpha, resid = best
    return alpha, lam, resid


def _smooth(a, window):
    """Media movil sobre el eje 0 (tiempo) con bordes replicados."""
    window = int(window)
    if window <= 1 or len(a) == 0:
        return a
    pad = window // 2
    ap = cp.pad(a, ((pad, pad), (0, 0)), mode='edge')
    return sum(ap[i:i + len(a)] for i in range(2 * pad + 1)) / (2 * pad + 1)


# ----------------------------------------------------------------------------
# Ajuste de una cabeza (conjunto de trayectorias + K)
# ----------------------------------------------------------------------------
def _fit_head(rows, raw, model, k, basis, max_len, with_cls=False):
    """Ajusta una cabeza con `rows` (y sus features crudos `raw`, (n, F))."""
    multimodal = model['mode'] == 'multimodal'
    qstd = np.asarray(model['stats']['qstd'], np.float32)
    weights, grid, sizes = _block_weights(), _ridge_grid(), block_sizes(model)
    n = len(rows)

    # features: centrar y balancear bloques
    raw = raw.astype(cp.float64)
    fmean = raw.mean(axis=0)
    Xc = raw - fmean
    scales, start = [], 0
    for name, size in sizes:
        blk = Xc[:, start:start + size]
        msq = float((blk ** 2).sum(axis=1).mean())
        sc = np.sqrt(weights[name] / max(msq, 1e-12))
        scales.append(cp.full((size,), sc, dtype=cp.float64))
        start += size
    fscale = cp.concatenate(scales)
    X = Xc * fscale

    G0 = X @ X.T
    scale0 = float(cp.trace(G0)) / n
    B = 1e4 * scale0                       # intercepto casi sin penalizar
    s, U = cp.linalg.eigh(G0 + B)
    lam_abs = scale0 * grid

    # targets con padding implicito (se repite el ultimo frame)
    Q = max_len - k
    rgb_t, jnt_t = [], []
    for row in rows:
        img = np.asarray(row['image'], np.float32)
        idx = np.minimum(np.arange(k, max_len), len(img) - 1)
        delta = cp.asarray(img[idx] - img[k - 1])              # (Q, IMAGE_DIM)
        rgb_t.append((delta @ basis).ravel())                  # (Q*r,)
        if multimodal:
            jt = np.asarray(row['joints'], np.float32)
            jnt_t.append(cp.asarray(((jt[idx] - jt[k - 1]) / qstd).ravel(), cp.float64))

    Y = cp.stack(rgb_t).astype(cp.float64)
    rgb_alpha, lam_rgb, res_rgb = _ridge_loo(U, s, Y, lam_abs)
    rgb_mean = B * rgb_alpha.sum(axis=0)   # sesgo = contribucion del intercepto

    head = dict(X=X.astype(cp.float32), fmean=fmean.astype(cp.float32), fscale=fscale.astype(cp.float32),
                rgb_alpha=rgb_alpha.astype(cp.float32), rgb_mean=rgb_mean.astype(cp.float32),
                basis=basis, q_values=np.arange(k, max_len, dtype=np.int32), Q=int(Q), class_heads={})
    info = dict(n=n, lam_rgb=lam_rgb, rmse_rgb=float(cp.sqrt(cp.mean(res_rgb ** 2))),
                base_rgb=float(cp.sqrt(cp.mean((Y - Y.mean(axis=0)) ** 2))))

    if multimodal:
        Yj = cp.stack(jnt_t)
        jnt_alpha, lam_j, res_j = _ridge_loo(U, s, Yj, lam_abs)
        jnt_mean = B * jnt_alpha.sum(axis=0)
        head['jnt_alpha'] = jnt_alpha.astype(cp.float32)
        head['jnt_mean'] = jnt_mean.astype(cp.float32)
        info.update(lam_jnt=lam_j, rmse_jnt=float(cp.sqrt(cp.mean(res_j ** 2))),
                    base_jnt=float(cp.sqrt(cp.mean((Yj - Yj.mean(axis=0)) ** 2))))

    if with_cls:
        labels = cp.asarray(np.array([int(row['label']) for row in rows], np.float64)[:, None])
        cls_alpha, lam_c, res_c = _ridge_loo(U, s, labels, lam_abs)
        cls_mean = B * cls_alpha.sum(axis=0)
        loo_pred = labels - res_c
        info.update(lam_cls=lam_c, loo_acc=float(cp.mean((loo_pred >= .5) == (labels >= .5))))
        head['cls_alpha'] = cls_alpha.astype(cp.float32)
        head['cls_mean'] = cls_mean.astype(cp.float32)
    return head, info


def _fmt(info):
    out = f"n={info['n']}; rgb-coef LOO={info['rmse_rgb']:.3f} (media={info['base_rgb']:.3f})"
    if 'rmse_jnt' in info:
        out += f"; joints LOO={info['rmse_jnt']:.3f} (media={info['base_jnt']:.3f})"
    return out


# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------
def build_model(train, N, training_epsilon=0, augmentation_count=0, seed=42):
    if training_epsilon != 0 or augmentation_count:
        raise NotImplementedError('This first multimodal version deliberately disables augmentation/noise.')
    from data_utils import compute_train_stats
    for row in train:
        _validate(row)
    stats = compute_train_stats(train)

    # Normalizacion del contexto: SOLO train.
    if _use_interaction():
        ctx = np.concatenate([_interaction_context(row) for row in train], axis=0)
        stats['imean'] = ctx.mean(axis=0).astype(np.float32)
        stats['istd'] = np.maximum(ctx.std(axis=0), 1e-6).astype(np.float32)

    model = new_model(int(N), int(seed), stats)

    rank = int(getattr(p, 'IMAGE_PCA_RANK', 48))
    model['bases']['all'] = _image_basis(train, rank, seed)
    model['basis'] = model['bases']['all']
    classes = sorted({int(row['label']) for row in train})
    min_rows = int(getattr(p, 'MIN_CLASS_ROWS', 8))
    cond = _class_conditional()
    if cond:
        for c in classes:
            rows_c = [row for row in train if int(row['label']) == c]
            if len(rows_c) >= min_rows:
                model['bases'][c] = _image_basis(rows_c, rank, seed)
    print(f"PCA de imagen: {model['basis'].shape[1]} componentes; clases={classes}; "
          f"condicionado={cond}; interaccion={model['use_interaction']}")

    for k in p.READOUT_K:
        rows = [row for row in train if len(row['image']) > k]
        if not rows:
            raise ValueError(f'No trajectories longer than K={k}')
        max_len = max(len(row['image']) for row in rows)
        raw = cp.stack([raw_feature(row, model, k) for row in rows])

        head, info = _fit_head(rows, raw, model, k, model['bases']['all'], max_len, with_cls=True)
        model['heads'][int(k)] = head
        model['accuracy'][int(k)] = info['loo_acc']
        print(f"K={k}: Q={max_len - k}; LOO acc={info['loo_acc']:.3f}; [general] {_fmt(info)}")

        if cond:
            for c in classes:
                ix = [i for i, row in enumerate(rows) if int(row['label']) == c]
                if len(ix) < min_rows or c not in model['bases']:
                    print(f'   clase {c}: solo {len(ix)} trayectorias, se usa la cabeza general')
                    continue
                sub, sinfo = _fit_head([rows[i] for i in ix], raw[ix], model, k,
                                       model['bases'][c], max_len, with_cls=False)
                head['class_heads'][c] = sub
                print(f'   [clase {c}] {_fmt(sinfo)}')
    return model


# ----------------------------------------------------------------------------
# Prediccion
# ----------------------------------------------------------------------------
def predict_trajectory(model, row, k, horizon=None, use_class=None):
    """Return (RGB predicted, joints predicted or None, push probability).

    use_class: fuerza la clase (0/1) para diagnostico; None = la decide el clasificador.
    """
    gen = model['heads'][int(k)]
    Q = gen['Q']
    horizon = min(len(row['image']) - k, Q) if horizon is None else int(horizon)
    if horizon < 0 or horizon > Q:
        raise ValueError('Invalid horizon')
    window = int(getattr(p, 'SMOOTH_WINDOW', 3))

    raw = raw_feature(row, model, k)
    score = float((gen['X'] @ _norm(raw, gen) @ gen['cls_alpha'] + gen['cls_mean']).item())

    head = gen
    if _class_conditional():
        c = int(score >= .5) if use_class is None else int(use_class)
        head = gen['class_heads'].get(c, gen)
    kx = head['X'] @ _norm(raw, head)                          # (n,)
    basis = head['basis']
    r = basis.shape[1]

    coef = (kx @ head['rgb_alpha'] + head['rgb_mean']).reshape(Q, r)
    coef = _smooth(coef, window)[:horizon]
    base_img = cp.asarray(row['image'][k - 1], cp.float32)
    delta_img = cp.clip(coef @ basis.T, -p.DELTA_CLIP, p.DELTA_CLIP)
    images = cp.asnumpy(cp.clip(base_img[None, :] + delta_img, 0, 1)).astype(np.float32)

    joints = None
    if model['mode'] == 'multimodal':
        jd = (kx @ head['jnt_alpha'] + head['jnt_mean']).reshape(Q, p.JOINT_DIM)
        jd = _smooth(jd, window)[:horizon]
        base_q = cp.asarray(row['joints'][k - 1], cp.float32)
        joints = cp.asnumpy(base_q[None, :] + jd * cp.asarray(model['stats']['qstd'])).astype(np.float32)
    return images, joints, score


# ----------------------------------------------------------------------------
# Guardar / cargar
# ----------------------------------------------------------------------------
def _save_head(payload, tag, head, multimodal):
    for name in HEAD_ARRAYS:
        payload[f'{name}_{tag}'] = cp.asnumpy(head[name])
    if multimodal:
        payload[f'jnt_alpha_{tag}'] = cp.asnumpy(head['jnt_alpha'])
        payload[f'jnt_mean_{tag}'] = cp.asnumpy(head['jnt_mean'])
    payload[f'qvals_{tag}'] = head['q_values']


def _load_head(d, tag, multimodal, basis):
    head = {name: cp.asarray(d[f'{name}_{tag}'], cp.float32) for name in HEAD_ARRAYS}
    if multimodal:
        head['jnt_alpha'] = cp.asarray(d[f'jnt_alpha_{tag}'], cp.float32)
        head['jnt_mean'] = cp.asarray(d[f'jnt_mean_{tag}'], cp.float32)
    head['q_values'] = np.asarray(d[f'qvals_{tag}'], np.int32)
    head['Q'] = int(len(head['q_values']))
    head['basis'] = basis
    head['class_heads'] = {}
    return head


def save_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    multimodal = model['mode'] == 'multimodal'
    payload = {key: cp.asnumpy(model[key]) for key in ('Win', 'W', 'bias')}
    for name, values in model['stats'].items():
        payload[f'stats_{name}'] = values
    for key, basis in model['bases'].items():
        payload[f'basis_{key}'] = cp.asnumpy(basis)
    for k, head in model['heads'].items():
        _save_head(payload, f'{k}', head, multimodal)
        payload[f'cls_alpha_{k}'] = cp.asnumpy(head['cls_alpha'])
        payload[f'cls_mean_{k}'] = cp.asnumpy(head['cls_mean'])
        payload[f'classes_{k}'] = np.asarray(sorted(head['class_heads']), np.int32)
        for c, sub in head['class_heads'].items():
            _save_head(payload, f'{k}_c{c}', sub, multimodal)
    np.savez_compressed(path, **payload, N=model['N'], seed=model['seed'],
                        mode=model['mode'], use_goal=int(model['use_goal']),
                        use_interaction=int(model['use_interaction']),
                        image_dim=model['image_dim'],
                        ks=np.asarray(sorted(model['heads']), np.int32),
                        architecture=ARCHITECTURE)


def load_model(path):
    with np.load(path, allow_pickle=False) as d:
        if str(d['architecture']) != ARCHITECTURE:
            raise ValueError(f"Modelo con arquitectura {d['architecture']}; se espera {ARCHITECTURE}. Reentrena.")
        mode = str(d['mode'])
        use_goal = bool(int(d['use_goal']))
        use_interaction = bool(int(d['use_interaction']))
        if (
            mode != p.EXPERIMENT_MODE
            or use_goal != p.USE_GOAL
            or use_interaction != _use_interaction()
            or int(d['image_dim']) != p.IMAGE_DIM
        ):
            raise ValueError('Model/config mode, goal, interaction-context, or image dimension mismatch')
        multimodal = mode == 'multimodal'
        stat_names = ['qmean', 'qstd', 'gmean', 'gstd']
        if use_interaction:
            stat_names += ['imean', 'istd']
        stats = {name: np.asarray(d[f'stats_{name}'], np.float32) for name in stat_names}
        model = {name: cp.asarray(d[name], cp.float32) for name in ('Win', 'W', 'bias')}
        bases = {}
        for key in d.files:
            if key.startswith('basis_'):
                tag = key[len('basis_'):]
                bases['all' if tag == 'all' else int(tag)] = cp.asarray(d[key], cp.float32)
        model.update(N=int(d['N']), seed=int(d['seed']), mode=mode,
                     use_goal=use_goal, use_interaction=use_interaction,
                     image_dim=int(d['image_dim']), stats=stats, heads={}, accuracy={},
                     bases=bases, basis=bases['all'])
        for k in d['ks']:
            k = int(k)
            head = _load_head(d, f'{k}', multimodal, bases['all'])
            head['cls_alpha'] = cp.asarray(d[f'cls_alpha_{k}'], cp.float32)
            head['cls_mean'] = cp.asarray(d[f'cls_mean_{k}'], cp.float32)
            for c in d[f'classes_{k}']:
                c = int(c)
                head['class_heads'][c] = _load_head(d, f'{k}_c{c}', multimodal, bases[c])
            model['heads'][k] = head
    return model