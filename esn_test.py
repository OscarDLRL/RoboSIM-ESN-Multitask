"""
CNN-latent Echo State Network for ManiSkill multimodal prediction.

Pipeline
--------
RGB -> CNN encoder -> compact visual latent z(t)
z(t) + joints(t) -> ESN -> future latent + future joints
future latent -> CNN decoder -> future RGB + agent/interation masks

Key difference from the old model:
The Ridge readout no longer predicts RGB pixels directly.
"""

from pathlib import Path
import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import parameters as p


# ============================================================
# Basic configuration
# ============================================================

def configuration():
    mode = p.EXPERIMENT_MODE
    if mode not in ("vision", "multimodal"):
        raise ValueError("Choose vision or multimodal")
    return mode, bool(p.USE_GOAL)


def mask_height():
    return int(p.MASK_HEIGHT)


def mask_width():
    return int(p.MASK_WIDTH)


def mask_dim():
    return mask_height() * mask_width()


def mask_threshold():
    return float(p.MASK_THRESHOLD)


def latent_dim():
    return int(p.VISUAL_LATENT_DIM)


def input_width():
    mode, goal = configuration()
    width = latent_dim()
    if mode == "multimodal":
        width += int(p.JOINT_DIM)
    if goal:
        width += int(p.GOAL_DIM)
    return width


def _torch_device():
    requested = str(getattr(p, "AE_DEVICE", "cuda"))
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


# ============================================================
# Masks
# ============================================================

def _resize_mask_nearest(sequence, out_h, out_w):
    """
    Occupancy-preserving binary resize.
    Downsampling uses max pooling so tiny objects do not disappear.
    """
    sequence = (np.asarray(sequence) > 0).astype(np.uint8)

    if sequence.ndim != 3:
        raise ValueError(f"Expected mask sequence (T,H,W), got {sequence.shape}")

    T, in_h, in_w = sequence.shape

    if in_h == out_h and in_w == out_w:
        return sequence

    if out_h <= in_h and out_w <= in_w:
        if in_h % out_h == 0 and in_w % out_w == 0:
            bh = in_h // out_h
            bw = in_w // out_w
            return (
                sequence.reshape(T, out_h, bh, out_w, bw)
                .max(axis=(2, 4))
                .astype(np.uint8)
            )

        result = np.zeros((T, out_h, out_w), dtype=np.uint8)
        y_edges = np.linspace(0, in_h, out_h + 1)
        x_edges = np.linspace(0, in_w, out_w + 1)

        for oy in range(out_h):
            y0 = int(np.floor(y_edges[oy]))
            y1 = max(int(np.ceil(y_edges[oy + 1])), y0 + 1)
            for ox in range(out_w):
                x0 = int(np.floor(x_edges[ox]))
                x1 = max(int(np.ceil(x_edges[ox + 1])), x0 + 1)
                result[:, oy, ox] = sequence[:, y0:y1, x0:x1].max(axis=(1, 2))

        return result

    yi = np.floor(np.arange(out_h) * in_h / out_h).astype(np.int64)
    xi = np.floor(np.arange(out_w) * in_w / out_w).astype(np.int64)
    yi = np.clip(yi, 0, in_h - 1)
    xi = np.clip(xi, 0, in_w - 1)
    return sequence[:, yi][:, :, xi]


def _mask_sequence(row, key):
    cache_key = f"__processed_{key}_{mask_height()}x{mask_width()}"

    if cache_key in row:
        return row[cache_key]

    if key not in row:
        raise KeyError(f"{row.get('id', '<unknown>')}: missing {key}")

    mask = np.asarray(row[key])

    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    if mask.ndim == 2:
        if mask.shape[1] != mask_dim():
            raise ValueError(f"{key}: bad flattened shape {mask.shape}")
        result = (mask > 0.5).astype(np.float32)
    elif mask.ndim == 3:
        result = (
            _resize_mask_nearest(mask, mask_height(), mask_width())
            .astype(np.float32)
            .reshape(len(mask), -1)
        )
    else:
        raise ValueError(f"{key}: invalid shape {mask.shape}")

    row[cache_key] = result
    return result


# ============================================================
# CNN visual representation
# ============================================================

class VisualAutoencoder(nn.Module):
    def __init__(self, channels, latent_size, mask_h, mask_w):
        super().__init__()

        if int(p.IMAGE_HEIGHT) != 64 or int(p.IMAGE_WIDTH) != 64:
            raise ValueError("This version expects RGB 64x64")

        self.channels = int(channels)
        self.latent_size = int(latent_size)
        self.mask_h = int(mask_h)
        self.mask_w = int(mask_w)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, 4, 2, 1),
            nn.ReLU(inplace=True),                    # 32x32
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),                    # 16x16
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),                    # 8x8
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU(inplace=True),                    # 4x4
        )

        self.to_latent = nn.Linear(128 * 4 * 4, self.latent_size)
        self.from_latent = nn.Linear(self.latent_size, 128 * 4 * 4)

        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.dec16 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.dec32 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

        # Semantic prediction at 32x32.
        self.mask_head = nn.Conv2d(32, 2, 3, padding=1)

        self.dec64 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.rgb_head = nn.Conv2d(16, self.channels, 3, padding=1)

    def encode(self, x):
        h = self.encoder(x)
        return self.to_latent(h.flatten(1))

    def decode(self, z):
        h = self.from_latent(z).reshape(-1, 128, 4, 4)
        h = self.dec8(h)
        h = self.dec16(h)
        h32 = self.dec32(h)

        mask_logits = self.mask_head(h32)
        if mask_logits.shape[-2:] != (self.mask_h, self.mask_w):
            mask_logits = F.interpolate(
                mask_logits,
                size=(self.mask_h, self.mask_w),
                mode="bilinear",
                align_corners=False,
            )

        h64 = self.dec64(h32)
        rgb = torch.sigmoid(self.rgb_head(h64))
        return rgb, mask_logits

    def forward(self, x):
        z = self.encode(x)
        rgb, masks = self.decode(z)
        return z, rgb, masks


class _FrameDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.index = [
            (ri, t)
            for ri, row in enumerate(rows)
            for t in range(len(row["image"]))
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        ri, t = self.index[index]
        row = self.rows[ri]

        c = int(p.IMAGE_CHANNELS)
        h = int(p.IMAGE_HEIGHT)
        w = int(p.IMAGE_WIDTH)

        rgb = (
            np.asarray(row["image"][t], np.float32)
            .reshape(h, w, c)
            .transpose(2, 0, 1)
        )

        agent = (
            _mask_sequence(row, "agent_mask")[t]
            .reshape(mask_height(), mask_width())
            .astype(np.float32)
        )

        interaction = (
            _mask_sequence(row, "interaction_mask")[t]
            .reshape(mask_height(), mask_width())
            .astype(np.float32)
        )

        return (
            torch.from_numpy(rgb),
            torch.from_numpy(agent[None]),
            torch.from_numpy(interaction[None]),
        )


def _dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum((1, 2, 3))
    denom = prob.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = (2 * inter + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def _train_autoencoder(train, seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    device = _torch_device()

    ae = VisualAutoencoder(
        int(p.IMAGE_CHANNELS),
        latent_dim(),
        mask_height(),
        mask_width(),
    ).to(device)

    dataset = _FrameDataset(train)
    generator = torch.Generator().manual_seed(int(seed))

    loader = DataLoader(
        dataset,
        batch_size=int(p.AE_BATCH_SIZE),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )

    optimizer = torch.optim.Adam(
        ae.parameters(),
        lr=float(p.AE_LR),
        weight_decay=float(p.AE_WEIGHT_DECAY),
    )

    print(
        f"Training CNN visual model: frames={len(dataset)}, "
        f"latent={latent_dim()}, epochs={p.AE_EPOCHS}, device={device}"
    )

    ae.train()

    for epoch in range(1, int(p.AE_EPOCHS) + 1):
        total = rgb_total = mask_total = 0.0
        n_batches = 0

        for rgb, agent, interaction in loader:
            rgb = rgb.to(device, non_blocking=True)
            agent = agent.to(device, non_blocking=True)
            interaction = interaction.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            _, pred_rgb, mask_logits = ae(rgb)

            agent64 = F.interpolate(
                agent,
                size=(int(p.IMAGE_HEIGHT), int(p.IMAGE_WIDTH)),
                mode="nearest",
            )
            interaction64 = F.interpolate(
                interaction,
                size=(int(p.IMAGE_HEIGHT), int(p.IMAGE_WIDTH)),
                mode="nearest",
            )

            weight = (
                1.0
                + float(p.AE_AGENT_RGB_WEIGHT) * agent64
                + float(p.AE_INTERACTION_RGB_WEIGHT) * interaction64
            )

            rgb_loss = (
                torch.abs(pred_rgb - rgb) * weight
            ).sum() / (weight.sum() * rgb.shape[1] + 1e-8)

            true_masks = torch.cat([agent, interaction], dim=1)
            bce = F.binary_cross_entropy_with_logits(mask_logits, true_masks)

            mask_loss = (
                float(p.AE_MASK_BCE_WEIGHT) * bce
                + _dice_loss(mask_logits[:, 0:1], agent)
                + float(p.AE_INTERACTION_DICE_WEIGHT)
                * _dice_loss(mask_logits[:, 1:2], interaction)
            )

            loss = rgb_loss + float(p.AE_MASK_LOSS_WEIGHT) * mask_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                ae.parameters(),
                float(p.AE_GRAD_CLIP),
            )
            optimizer.step()

            total += float(loss.detach().cpu())
            rgb_total += float(rgb_loss.detach().cpu())
            mask_total += float(mask_loss.detach().cpu())
            n_batches += 1

        if (
            epoch == 1
            or epoch == int(p.AE_EPOCHS)
            or epoch % int(p.AE_PRINT_EVERY) == 0
        ):
            print(
                f"AE epoch {epoch:03d}/{p.AE_EPOCHS}: "
                f"loss={total/n_batches:.5f} "
                f"rgb={rgb_total/n_batches:.5f} "
                f"mask={mask_total/n_batches:.5f}"
            )

    ae.eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)

    return ae


def _images_tensor(images):
    images = np.asarray(images, np.float32)
    c = int(p.IMAGE_CHANNELS)
    h = int(p.IMAGE_HEIGHT)
    w = int(p.IMAGE_WIDTH)
    return torch.from_numpy(
        images.reshape(-1, h, w, c).transpose(0, 3, 1, 2)
    )


def _encode_images(ae, images):
    device = next(ae.parameters()).device
    x = _images_tensor(images)
    outputs = []

    with torch.no_grad():
        for start in range(0, len(x), int(p.AE_INFERENCE_BATCH_SIZE)):
            xb = x[start:start + int(p.AE_INFERENCE_BATCH_SIZE)].to(
                device, non_blocking=True
            )
            outputs.append(
                ae.encode(xb).detach().cpu().numpy().astype(np.float32)
            )

    return np.concatenate(outputs, axis=0)


def _decode_latents(ae, latents):
    device = next(ae.parameters()).device
    latents = np.asarray(latents, np.float32)

    rgb_out = []
    agent_out = []
    interaction_out = []

    with torch.no_grad():
        for start in range(0, len(latents), int(p.AE_INFERENCE_BATCH_SIZE)):
            z = torch.from_numpy(
                latents[start:start + int(p.AE_INFERENCE_BATCH_SIZE)]
            ).to(device, non_blocking=True)

            rgb, logits = ae.decode(z)
            masks = torch.sigmoid(logits)

            rgb_out.append(rgb.detach().cpu().numpy().astype(np.float32))
            agent_out.append(masks[:, 0].detach().cpu().numpy().astype(np.float32))
            interaction_out.append(masks[:, 1].detach().cpu().numpy().astype(np.float32))

    rgb = np.concatenate(rgb_out, axis=0)
    rgb = rgb.transpose(0, 2, 3, 1).reshape(len(rgb), -1)

    return (
        rgb,
        np.concatenate(agent_out, axis=0),
        np.concatenate(interaction_out, axis=0),
    )


# ============================================================
# Latent caching / normalization
# ============================================================

def _raw_latent_sequence(row, model):
    key = model["_latent_cache_key"]
    if key not in row:
        row[key] = _encode_images(model["ae"], row["image"])
    return row[key]


def _normalized_latent_sequence(row, model):
    z = _raw_latent_sequence(row, model)
    return (z - model["stats"]["zmean"]) / model["stats"]["zstd"]


def _compute_latent_stats(train, model):
    z = np.concatenate(
        [_raw_latent_sequence(row, model) for row in train],
        axis=0,
    )

    return (
        z.mean(axis=0).astype(np.float32),
        np.maximum(z.std(axis=0), 1e-5).astype(np.float32),
    )


# ============================================================
# ESN
# ============================================================

def _validate(row):
    image = np.asarray(row["image"])

    if image.ndim != 2 or image.shape[1] != int(p.IMAGE_DIM):
        raise ValueError(
            f"{row['id']}: expected image (T,{p.IMAGE_DIM}), got {image.shape}"
        )

    if len(image) != len(row["joints"]):
        raise ValueError(f"{row['id']}: image/joint frame mismatch")

    if np.asarray(row["goal"]).shape != (3,):
        raise ValueError(f"{row['id']}: expected goal XYZ shape (3,)")

    if len(_mask_sequence(row, "agent_mask")) != len(image):
        raise ValueError(f"{row['id']}: agent mask mismatch")

    if len(_mask_sequence(row, "interaction_mask")) != len(image):
        raise ValueError(f"{row['id']}: interaction mask mismatch")


def new_model(N, seed, stats, ae):
    rng = cp.random.RandomState(int(seed))
    dim = input_width()

    Win = rng.uniform(
        -p.INPUT_SCALE, p.INPUT_SCALE, (N, dim)
    ).astype(cp.float32)

    bias = rng.uniform(
        -p.INPUT_SCALE, p.INPUT_SCALE, (N,)
    ).astype(cp.float32)

    W = rng.uniform(-1, 1, (N, N)).astype(cp.float32)

    radius = float(cp.max(cp.abs(cp.linalg.eigvals(W))).get())
    W *= cp.float32(p.SPECTRAL_RADIUS / max(radius, 1e-12))

    return dict(
        Win=Win,
        W=W,
        bias=bias,
        N=int(N),
        seed=int(seed),
        mode=p.EXPERIMENT_MODE,
        use_goal=bool(p.USE_GOAL),
        image_dim=int(p.IMAGE_DIM),
        latent_dim=latent_dim(),
        mask_height=mask_height(),
        mask_width=mask_width(),
        mask_dim=mask_dim(),
        stats=stats,
        ae=ae,
        heads={},
        accuracy={},
        _latent_cache_key=f"__latent_{id(ae)}",
    )


def prepare_observation(row, model, k):
    z = cp.asarray(
        _normalized_latent_sequence(row, model)[:k],
        cp.float32,
    )

    modalities = [
        cp.float32(p.LATENT_INPUT_SCALE) * (z[1:] - z[:-1])
    ]

    if model["mode"] == "multimodal":
        raw_q = cp.asarray(row["joints"][:k], cp.float32)
        mean = cp.asarray(model["stats"]["qmean"], cp.float32)
        std = cp.asarray(model["stats"]["qstd"], cp.float32)
        q = (raw_q - mean) / std

        modalities.append(
            cp.float32(p.JOINT_INPUT_SCALE) * (q[1:] - q[:-1])
        )

    if model["use_goal"]:
        goal = (
            cp.asarray(row["goal"], cp.float32)
            - cp.asarray(model["stats"]["gmean"], cp.float32)
        ) / cp.asarray(model["stats"]["gstd"], cp.float32)

        modalities.append(
            cp.broadcast_to(
                cp.float32(p.GOAL_INPUT_SCALE) * goal,
                (k - 1, int(p.GOAL_DIM)),
            )
        )

    return cp.concatenate(modalities, axis=1)


def feature(row, model, k):
    obs = prepare_observation(row, model, k)

    state = cp.zeros((model["N"],), dtype=cp.float32)

    for t in range(len(obs)):
        candidate = cp.tanh(
            model["bias"]
            + model["Win"] @ obs[t]
            + model["W"] @ state
        )

        state = (
            (1 - p.LEAK_RATE) * state
            + p.LEAK_RATE * candidate
        )

    z = cp.asarray(
        _normalized_latent_sequence(row, model)[k - 1],
        cp.float32,
    )

    base = [
        cp.ones((1,), dtype=cp.float32),
        state,
        cp.float32(p.LATENT_INPUT_SCALE) * z,
    ]

    if model["mode"] == "multimodal":
        q = (
            cp.asarray(row["joints"][k - 1], cp.float32)
            - cp.asarray(model["stats"]["qmean"], cp.float32)
        ) / cp.asarray(model["stats"]["qstd"], cp.float32)

        base.append(cp.float32(p.JOINT_INPUT_SCALE) * q)

    if model["use_goal"]:
        goal = (
            cp.asarray(row["goal"], cp.float32)
            - cp.asarray(model["stats"]["gmean"], cp.float32)
        ) / cp.asarray(model["stats"]["gstd"], cp.float32)

        base.append(cp.float32(p.GOAL_INPUT_SCALE) * goal)

    return cp.concatenate(base)


def _target(row, k, q, model):
    z = _normalized_latent_sequence(row, model)

    latent_delta = (z[q] - z[k - 1]).astype(np.float32)

    if model["mode"] == "vision":
        return latent_delta

    joint_delta = (
        row["joints"][q] - row["joints"][k - 1]
    ) / model["stats"]["qstd"]

    return np.concatenate(
        [latent_delta, joint_delta.astype(np.float32)]
    )


def build_model(train, N, training_epsilon=0, augmentation_count=0, seed=42):
    if training_epsilon != 0 or augmentation_count:
        raise NotImplementedError(
            "This first latent-CNN experiment disables ESN augmentation/noise."
        )

    from data_utils import compute_train_stats

    for row in train:
        _validate(row)

    # 1. Learn visual representation using TRAIN only.
    ae = _train_autoencoder(train, seed)

    # 2. Train-only joint/goal stats.
    stats = compute_train_stats(train)

    # 3. Compute train-only latent mean/std.
    temp_model = {
        "ae": ae,
        "_latent_cache_key": f"__latent_temp_{id(ae)}",
    }

    zmean, zstd = _compute_latent_stats(train, temp_model)
    stats["zmean"] = zmean
    stats["zstd"] = zstd

    # 4. Create reservoir.
    model = new_model(int(N), int(seed), stats, ae)

    # Reuse latents already encoded above.
    for row in train:
        if temp_model["_latent_cache_key"] in row:
            row[model["_latent_cache_key"]] = row[temp_model["_latent_cache_key"]]

    # 5. Direct future latent/joint readouts.
    for k in p.READOUT_K:
        rows = [r for r in train if len(r["image"]) > k]

        if not rows:
            raise ValueError(f"No trajectories longer than K={k}")

        X = cp.stack(
            [feature(row, model, k) for row in rows]
        ).astype(cp.float32)

        cls_y = cp.asarray(
            np.asarray([r["label"] for r in rows], np.float64)[:, None]
        )

        X64 = X.astype(cp.float64)
        gram = (
            X64 @ X64.T
            + p.READOUT_RIDGE * cp.eye(len(rows), dtype=cp.float64)
        )

        cls_alpha = cp.linalg.solve(
            gram, cls_y
        ).astype(cp.float32)

        fitted = (
            ((X @ X.T) @ cls_alpha).ravel().get() >= 0.5
        )

        cls_accuracy = float(
            np.mean(fitted == np.asarray([r["label"] for r in rows]))
        )

        max_len = max(len(row["image"]) for row in rows)
        alphas = []
        indices = []

        for q in range(k, max_len):
            ix = [
                j
                for j, row in enumerate(rows)
                if len(row["image"]) > q
            ]

            Xq = X[ix].astype(cp.float64)

            Y = cp.asarray(
                np.stack(
                    [_target(rows[j], k, q, model) for j in ix]
                ),
                cp.float64,
            )

            A = (
                Xq @ Xq.T
                + p.READOUT_RIDGE * cp.eye(len(ix), dtype=cp.float64)
            )

            alphas.append(
                cp.linalg.solve(A, Y).astype(cp.float32)
            )

            indices.append(
                np.asarray(ix, dtype=np.int32)
            )

        model["heads"][int(k)] = dict(
            X=X,
            cls_alpha=cls_alpha,
            q_values=np.arange(k, max_len, dtype=np.int32),
            alphas=alphas,
            indices=indices,
        )

        model["accuracy"][int(k)] = cls_accuracy

        print(
            f"K={k}: n={len(rows)}; final frame index={max_len-1}; "
            f"training action accuracy={cls_accuracy:.3f}; "
            f"visual_latent={latent_dim()}"
        )

    return model


# ============================================================
# Prediction
# ============================================================

def predict_trajectory(
    model,
    row,
    k,
    horizon=None,
    return_masks=False,
    binary_masks=False,
):
    head = model["heads"][int(k)]

    max_h = len(head["q_values"])

    horizon = (
        min(len(row["image"]) - k, max_h)
        if horizon is None
        else int(horizon)
    )

    if horizon < 0 or horizon > max_h:
        raise ValueError("Invalid horizon")

    x = feature(row, model, k)

    score = float(
        (x @ head["X"].T @ head["cls_alpha"]).item()
    )

    base_z = cp.asarray(
        _normalized_latent_sequence(row, model)[k - 1],
        cp.float32,
    )

    base_q = cp.asarray(
        row["joints"][k - 1],
        cp.float32,
    )

    z_future = []
    q_future = []

    for step in range(horizon):
        ix = head["indices"][step]

        delta = (
            x
            @ head["X"][ix].T
            @ head["alphas"][step]
        )

        dz = delta[:model["latent_dim"]]

        dz = cp.clip(
            dz,
            -float(p.LATENT_DELTA_CLIP),
            float(p.LATENT_DELTA_CLIP),
        )

        z_future.append(base_z + dz)

        if model["mode"] == "multimodal":
            dq = delta[
                model["latent_dim"]:
                model["latent_dim"] + int(p.JOINT_DIM)
            ]

            q_future.append(
                base_q
                + dq
                * cp.asarray(model["stats"]["qstd"], cp.float32)
            )

    if z_future:
        z_norm = cp.asnumpy(cp.stack(z_future))

        z_raw = (
            z_norm * model["stats"]["zstd"][None]
            + model["stats"]["zmean"][None]
        ).astype(np.float32)

        images, agent_masks, interaction_masks = _decode_latents(
            model["ae"],
            z_raw,
        )
    else:
        images = np.empty((0, int(p.IMAGE_DIM)), np.float32)
        agent_masks = np.empty(
            (0, model["mask_height"], model["mask_width"]),
            np.float32,
        )
        interaction_masks = np.empty_like(agent_masks)

    if model["mode"] == "multimodal":
        joints = (
            cp.asnumpy(cp.stack(q_future))
            if q_future
            else np.empty((0, int(p.JOINT_DIM)), np.float32)
        )
    else:
        joints = None

    if binary_masks:
        th = mask_threshold()
        agent_masks = (agent_masks >= th).astype(np.uint8)
        interaction_masks = (interaction_masks >= th).astype(np.uint8)

    if return_masks:
        return (
            images,
            joints,
            score,
            agent_masks,
            interaction_masks,
        )

    return images, joints, score


# ============================================================
# Persistence
# ============================================================

def _safe_key(name):
    return name.replace(".", "__DOT__")


def _unsafe_key(name):
    return name.replace("__DOT__", ".")


def save_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        name: cp.asnumpy(model[name])
        for name in ("Win", "W", "bias")
    }

    stat_names = sorted(model["stats"].keys())
    payload["stat_names"] = np.asarray(stat_names, dtype="U32")

    for name in stat_names:
        payload[f"stats_{name}"] = np.asarray(
            model["stats"][name],
            np.float32,
        )

    ae_keys = []

    for name, tensor in model["ae"].state_dict().items():
        safe = _safe_key(name)
        ae_keys.append(safe)
        payload[f"ae_{safe}"] = tensor.detach().cpu().numpy()

    payload["ae_keys"] = np.asarray(ae_keys, dtype="U128")

    for k, head in model["heads"].items():
        payload[f"X_{k}"] = cp.asnumpy(head["X"])
        payload[f"cls_{k}"] = cp.asnumpy(head["cls_alpha"])
        payload[f"qvals_{k}"] = head["q_values"]

        for i, (alpha, ix) in enumerate(
            zip(head["alphas"], head["indices"])
        ):
            payload[f"a_{k}_{i}"] = cp.asnumpy(alpha)
            payload[f"ix_{k}_{i}"] = ix

    np.savez_compressed(
        path,
        **payload,
        N=model["N"],
        seed=model["seed"],
        mode=model["mode"],
        use_goal=int(model["use_goal"]),
        image_dim=model["image_dim"],
        latent_dim=model["latent_dim"],
        mask_height=model["mask_height"],
        mask_width=model["mask_width"],
        ks=np.asarray(sorted(model["heads"]), np.int32),
        architecture="latent_cnn_esn_v1",
    )


def load_model(path):
    device = _torch_device()

    with np.load(path, allow_pickle=False) as d:
        architecture = str(d["architecture"].item())

        if architecture != "latent_cnn_esn_v1":
            raise ValueError(
                f"Expected latent_cnn_esn_v1, got {architecture}"
            )

        mode = str(d["mode"].item())
        use_goal = bool(int(d["use_goal"]))
        image_dim = int(d["image_dim"])
        stored_latent = int(d["latent_dim"])
        stored_mask_h = int(d["mask_height"])
        stored_mask_w = int(d["mask_width"])

        if (
            mode != p.EXPERIMENT_MODE
            or use_goal != bool(p.USE_GOAL)
            or image_dim != int(p.IMAGE_DIM)
            or stored_latent != latent_dim()
            or stored_mask_h != mask_height()
            or stored_mask_w != mask_width()
        ):
            raise ValueError("Model/config mismatch")

        stat_names = [str(x) for x in d["stat_names"]]

        stats = {
            name: np.asarray(d[f"stats_{name}"], np.float32)
            for name in stat_names
        }

        ae = VisualAutoencoder(
            int(p.IMAGE_CHANNELS),
            stored_latent,
            stored_mask_h,
            stored_mask_w,
        ).to(device)

        state = {}

        for safe_raw in d["ae_keys"]:
            safe = str(safe_raw)
            state[_unsafe_key(safe)] = torch.from_numpy(
                np.asarray(d[f"ae_{safe}"])
            )

        ae.load_state_dict(state)
        ae.eval()

        for parameter in ae.parameters():
            parameter.requires_grad_(False)

        model = {
            name: cp.asarray(d[name], cp.float32)
            for name in ("Win", "W", "bias")
        }

        model.update(
            N=int(d["N"]),
            seed=int(d["seed"]),
            mode=mode,
            use_goal=use_goal,
            image_dim=image_dim,
            latent_dim=stored_latent,
            mask_height=stored_mask_h,
            mask_width=stored_mask_w,
            mask_dim=stored_mask_h * stored_mask_w,
            stats=stats,
            ae=ae,
            heads={},
            accuracy={},
            _latent_cache_key=f"__latent_loaded_{id(ae)}",
        )

        for k_raw in d["ks"]:
            k = int(k_raw)
            qvals = np.asarray(d[f"qvals_{k}"], np.int32)

            model["heads"][k] = dict(
                X=cp.asarray(d[f"X_{k}"], cp.float32),
                cls_alpha=cp.asarray(d[f"cls_{k}"], cp.float32),
                q_values=qvals,
                alphas=[
                    cp.asarray(d[f"a_{k}_{i}"], cp.float32)
                    for i in range(len(qvals))
                ],
                indices=[
                    np.asarray(d[f"ix_{k}_{i}"], np.int32)
                    for i in range(len(qvals))
                ],
            )

    return model