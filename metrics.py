import numpy as np

METRIC_VERSION = 'motion_v1'
PHASE_METRIC_VERSION = 'direct_forecast_phases_v2'


def frame_deltas(frames, previous):
    frames = np.asarray(frames, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32).reshape(1, -1)
    if frames.ndim != 2 or frames.shape[1] != previous.shape[1]:
        raise ValueError('frames must have shape (T, D); previous must contain D pixels')
    return np.diff(np.concatenate((previous, frames), axis=0), axis=0)


def motion_mask(delta, threshold):
    return np.abs(delta) >= threshold


def safe_div(a, b):
    return float(a / b) if b > 0 else 0.0


def masked_rmse(error, mask):
    values = error[mask]
    return float(np.sqrt(np.mean(values**2))) if values.size else 0.0


def masked_mae(error, mask):
    values = np.abs(error[mask])
    return float(np.mean(values)) if values.size else 0.0


def centroid(weights):
    total = weights.sum()
    if total <= 0:
        return None
    h, w = weights.shape
    cy = float((weights.sum(axis=1) * np.arange(h)).sum() / total)
    cx = float((weights.sum(axis=0) * np.arange(w)).sum() / total)
    return np.array([cy, cx], dtype=np.float64)


def motion_centroid_error(true_delta, pred_delta, height, width, threshold):
    true_maps = np.abs(true_delta).reshape(-1, height, width)
    pred_maps = np.abs(pred_delta).reshape(-1, height, width)
    diagonal = float(np.hypot(height, width))
    errors = []
    for true_frame, pred_frame in zip(true_maps, pred_maps):
        tc = centroid(np.where(true_frame >= threshold, true_frame, 0))
        if tc is None:
            continue
        pc = centroid(np.where(pred_frame >= threshold, pred_frame, 0))
        errors.append(diagonal if pc is None else float(np.linalg.norm(tc - pc)))
    return float(np.mean(errors)) if errors else 0.0


def phase_metrics(true_frames, pred_frames, true_delta, pred_delta, threshold):
    """Evaluate any forecast phase; no assumption of recurrent feedback."""
    if len(true_frames) == 0:
        return dict.fromkeys(('image_rmse', 'motion_rmse', 'motion_skill',
                               'motion_f1', 'ghost_energy_ratio'), np.nan)
    true_motion = motion_mask(true_delta, threshold)
    pred_motion = motion_mask(pred_delta, threshold)
    err = pred_delta - true_delta
    motion_rmse = masked_rmse(err, true_motion)
    baseline_rmse = masked_rmse(true_delta, true_motion)
    tp = np.count_nonzero(true_motion & pred_motion)
    fp = np.count_nonzero(~true_motion & pred_motion)
    fn = np.count_nonzero(true_motion & ~pred_motion)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    true_energy = float(np.abs(true_delta[true_motion]).sum())
    ghost_energy = float(np.abs(pred_delta[pred_motion & ~true_motion]).sum())
    return {
        'image_rmse': float(np.sqrt(np.mean((pred_frames - true_frames)**2))),
        'motion_rmse': motion_rmse,
        'motion_skill': 1.0 - motion_rmse / baseline_rmse if baseline_rmse > 0 else 0.0,
        'motion_f1': f1,
        'ghost_energy_ratio': safe_div(ghost_energy, true_energy),
    }


def _evaluate_sequence_gray(true_frames, pred_frames, previous_frame, height, width, threshold=0.05):
    true_frames = np.asarray(true_frames, dtype=np.float32)
    pred_frames = np.asarray(pred_frames, dtype=np.float32)
    if true_frames.shape != pred_frames.shape or true_frames.ndim != 2:
        raise ValueError('true_frames and pred_frames must be equal-shape (T, D) arrays')
    if not len(true_frames):
        raise ValueError('evaluate_sequence expects at least one forecast frame')
    if true_frames.shape[1] != height * width:
        raise ValueError('image dimensions do not match height * width (grayscale expected)')
    true_delta = frame_deltas(true_frames, previous_frame)
    pred_delta = frame_deltas(pred_frames, previous_frame)
    error = pred_delta - true_delta
    true_motion = motion_mask(true_delta, threshold)
    pred_motion = motion_mask(pred_delta, threshold)
    tp = np.count_nonzero(true_motion & pred_motion)
    fp = np.count_nonzero(~true_motion & pred_motion)
    fn = np.count_nonzero(true_motion & ~pred_motion)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    baseline_rmse = masked_rmse(true_delta, true_motion)
    motion_rmse = masked_rmse(error, true_motion)
    true_energy = float(np.abs(true_delta[true_motion]).sum())
    ghost_energy = float(np.abs(pred_delta[pred_motion & ~true_motion]).sum())
    image_error = pred_frames - true_frames
    first = phase_metrics(true_frames[:1], pred_frames[:1], true_delta[:1], pred_delta[:1], threshold)
    rest = phase_metrics(true_frames[1:], pred_frames[1:], true_delta[1:], pred_delta[1:], threshold)
    result = {
        'metric_version': METRIC_VERSION,
        'phase_metric_version': PHASE_METRIC_VERSION,
        'motion_threshold': float(threshold),
        'motion_rmse': motion_rmse,
        'motion_mae': masked_mae(error, true_motion),
        'motion_skill': 1.0 - motion_rmse / baseline_rmse if baseline_rmse > 0 else 0.0,
        'motion_precision': precision,
        'motion_recall': recall,
        'motion_f1': safe_div(2 * precision * recall, precision + recall),
        'ghost_energy_ratio': safe_div(ghost_energy, true_energy),
        'motion_centroid_error_px': motion_centroid_error(true_delta, pred_delta, height, width, threshold),
        'true_motion_fraction': float(true_motion.mean()),
        'pred_motion_fraction': float(pred_motion.mean()),
        'image_rmse': float(np.sqrt(np.mean(image_error**2))),
        'image_mae': float(np.mean(np.abs(image_error))),
    }
    for prefix, phase in (('first', first), ('autonomous', rest)):
        # 'autonomous' is retained as a legacy CSV key ONLY; these are
        # direct future forecasts, NOT a feedback-driven rollout.
        for key in ('image_rmse', 'motion_rmse', 'motion_skill', 'motion_f1', 'ghost_energy_ratio'):
            result[f'{prefix}_{key}'] = phase[key]
    return result


def _motion_rmse_horizon_gray(true_frames, pred_frames, previous_frame, threshold=0.05):
    true_delta = frame_deltas(true_frames, previous_frame)
    pred_delta = frame_deltas(pred_frames, previous_frame)
    esn, baseline = [], []
    for actual, predicted in zip(true_delta, pred_delta):
        mask = motion_mask(actual, threshold)
        if not mask.any():
            esn.append(np.nan)
            baseline.append(np.nan)
        else:
            esn.append(float(np.sqrt(np.mean((predicted[mask] - actual[mask])**2))))
            baseline.append(float(np.sqrt(np.mean(actual[mask]**2))))
    return np.asarray(esn), np.asarray(baseline)


def freeze_ratio_horizon(true_frames, pred_frames, previous_frame, eps=1e-8):
    previous = np.asarray(previous_frame, dtype=np.float32).reshape(1, -1)
    true_change = np.mean(np.abs(np.asarray(true_frames) - previous), axis=1)
    pred_change = np.mean(np.abs(np.asarray(pred_frames) - previous), axis=1)
    return pred_change / (true_change + eps), true_change, pred_change


def _motion_luminance(frames, height, width):
    """RGB HWC flattened -> grayscale movement signal; grayscale passes through."""
    a = np.asarray(frames, dtype=np.float32)
    d = height * width
    if a.shape[-1] == d:
        return a
    if a.shape[-1] != 3 * d:
        raise ValueError(f"Expected {d} grayscale or {3*d} RGB values, got {a.shape[-1]}")
    shape = a.shape[:-1] + (height, width, 3)
    rgb = a.reshape(shape)
    gray = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    return gray.reshape(a.shape[:-1] + (d,))


def evaluate_sequence(true_frames, pred_frames, previous_frame, height, width, threshold=0.05):
    """Motion metrics on RGB luminance; image RMSE/MAE on all RGB channels."""
    true = np.asarray(true_frames, dtype=np.float32)
    pred = np.asarray(pred_frames, dtype=np.float32)
    previous = np.asarray(previous_frame, dtype=np.float32)
    if true.shape != pred.shape:
        raise ValueError("True and predicted image arrays have different shapes")
    yt = _motion_luminance(true, height, width)
    yp = _motion_luminance(pred, height, width)
    yprev = _motion_luminance(previous, height, width)
    result = _evaluate_sequence_gray(yt, yp, yprev, height, width, threshold)
    err = pred - true
    result["image_rmse"] = float(np.sqrt(np.mean(err ** 2)))
    result["image_mae"] = float(np.mean(np.abs(err)))
    result["first_image_rmse"] = float(np.sqrt(np.mean(err[:1] ** 2)))
    result["autonomous_image_rmse"] = (
        float(np.sqrt(np.mean(err[1:] ** 2))) if len(err) > 1 else np.nan
    )
    result["metric_version"] = "motion_luma_rgb_v1" if true.shape[1] == height * width * 3 else METRIC_VERSION
    return result


def motion_rmse_horizon(true_frames, pred_frames, previous_frame, threshold=0.05, height=None, width=None):
    true = np.asarray(true_frames)
    if height is not None and width is not None:
        true = _motion_luminance(true, height, width)
        pred_frames = _motion_luminance(pred_frames, height, width)
        previous_frame = _motion_luminance(previous_frame, height, width)
    return _motion_rmse_horizon_gray(true, pred_frames, previous_frame, threshold)

def pixel_rmse(true, pred):
    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    if true.shape != pred.shape:
        raise ValueError("Prediction and ground truth shapes differ")

    return float(np.sqrt(np.mean((pred - true) ** 2)))


def delta_image_rmse(true, pred, previous):
    true_delta = frame_deltas(true, previous)
    pred_delta = frame_deltas(pred, previous)

    return float(
        np.sqrt(np.mean((pred_delta - true_delta) ** 2))
    )


def moving_region_rmse(
    true,
    pred,
    previous,
    height,
    width,
    threshold=0.08
):
    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    true_gray = _motion_luminance(true, height, width)
    pred_gray = _motion_luminance(pred, height, width)
    previous_gray = _motion_luminance(
        np.asarray(previous, dtype=np.float32),
        height,
        width
    )

    true_delta = frame_deltas(true_gray, previous_gray)
    pred_delta = frame_deltas(pred_gray, previous_gray)

    mask = (
        (np.abs(true_delta) >= threshold)
        |
        (np.abs(pred_delta) >= threshold)
    )

    if not np.any(mask):
        return 0.0

    channels = true.shape[1] // (height * width)

    mask = np.repeat(
        mask.reshape(-1, height * width, 1),
        channels,
        axis=2
    ).reshape(true.shape)

    return float(
        np.sqrt(np.mean((pred[mask] - true[mask]) ** 2))
    )

def prediction_metrics(
    true,
    pred,
    previous,
    height,
    width,
    threshold=0.08
):
    """
    Evaluate RGB prediction quality.

    Returns:
        pixel_rmse
        delta_image_rmse
        moving_region_rmse
        motion_f1
    """

    import numpy as np

    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32).reshape(1, -1)

    if true.shape != pred.shape:
        raise ValueError("Ground truth and prediction shapes differ")

    if true.ndim != 2 or previous.shape[1] != true.shape[1]:
        raise ValueError("Invalid image dimensions")

    channels = true.shape[1] // (height * width)

    if channels not in (1, 3):
        raise ValueError("Expected grayscale or RGB images")

    # 1. Overall pixel RMSE
    pixel_rmse = float(
        np.sqrt(np.mean((pred - true) ** 2))
    )

    # 2. Consecutive frame differences
    true_delta = np.diff(
        np.concatenate([previous, true], axis=0),
        axis=0
    )

    pred_delta = np.diff(
        np.concatenate([previous, pred], axis=0),
        axis=0
    )

    delta_image_rmse = float(
        np.sqrt(np.mean((pred_delta - true_delta) ** 2))
    )

    # 3. Convert RGB differences to luminance
    def luminance(delta):
        images = delta.reshape(-1, height, width, channels)

        if channels == 1:
            return images[..., 0]

        return (
            0.2126 * images[..., 0]
            + 0.7152 * images[..., 1]
            + 0.0722 * images[..., 2]
        )

    true_motion = np.abs(luminance(true_delta))
    pred_motion = np.abs(luminance(pred_delta))

    true_mask = true_motion >= threshold
    pred_mask = pred_motion >= threshold

    # 4. Moving region: actual OR predicted motion
    union = true_mask | pred_mask

    if union.any():

        pixel_mask = np.broadcast_to(
            union[..., None],
            (len(true), height, width, channels)
        )

        image_error = (
            pred - true
        ).reshape(-1, height, width, channels)

        moving_region_rmse = float(
            np.sqrt(np.mean(image_error[pixel_mask] ** 2))
        )

    else:
        moving_region_rmse = 0.0

    # 5. Motion F1
    tp = np.count_nonzero(true_mask & pred_mask)
    fp = np.count_nonzero(~true_mask & pred_mask)
    fn = np.count_nonzero(true_mask & ~pred_mask)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    motion_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    return {
        "pixel_rmse": pixel_rmse,
        "delta_image_rmse": delta_image_rmse,
        "moving_region_rmse": moving_region_rmse,
        "motion_f1": float(motion_f1),
    }

def per_frame_metrics(
    true,
    pred,
    previous,
    height,
    width
):
    """
    Calculate prediction errors for each future frame.

    Returns:
        pixel_rmse: (T,)
        delta_rmse: (T,)

    RGB values are expected in [0, 1].
    """

    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32).reshape(1, -1)

    if true.shape != pred.shape:
        raise ValueError(
            "True and predicted images must have identical shapes"
        )

    if true.ndim != 2:
        raise ValueError("Expected images with shape (T, D)")

    if true.shape[1] not in (height * width, height * width * 3):
        raise ValueError("Invalid image dimensions")

    if previous.shape[1] != true.shape[1]:
        raise ValueError("Previous image dimension mismatch")

    if len(true) == 0:
        return np.empty(0), np.empty(0)

    # RGB reconstruction error at each future frame.
    pixel_rmse = np.sqrt(
        np.mean((pred - true) ** 2, axis=1)
    )

    # Consecutive image differences.
    # First future frame is compared against last observed frame.
    true_delta = np.diff(
        np.concatenate([previous, true], axis=0),
        axis=0
    )

    pred_delta = np.diff(
        np.concatenate([previous, pred], axis=0),
        axis=0
    )

    # Delta prediction error at each future frame.
    delta_rmse = np.sqrt(
        np.mean((pred_delta - true_delta) ** 2, axis=1)
    )

    return pixel_rmse, delta_rmse