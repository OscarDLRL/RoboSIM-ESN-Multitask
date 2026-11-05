
from pathlib import Path
import os
import numpy as np


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

MODEL_NAME = "motion_only_hybrid_v13"
ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "datasets"
    / "maniskill_intent_15cm"
    / "multimodal_intent_v1.h5"
)

RESULTS_DIR = (
    ROOT
    / "Results_esn_multitask"
    / "intent_diagonal_right_15cm"
    / MODEL_NAME
)


# -----------------------------------------------------------------------------
# Final architecture
# -----------------------------------------------------------------------------

TRAIN_SPLITS = ("train", "validation")
EXPERIMENT_MODE = "dropout"

# v11 is fixed: no goal, no interaction context, joint prediction ON,
# class-conditional heads ON.
JOINT_TARGET_MODE = "delta_from_qK_with_visual_anchor"


# -----------------------------------------------------------------------------
# Modalities
# -----------------------------------------------------------------------------

MODALITY_DROPOUT_PATTERN = (
    ("full", 2),
    ("vision", 1),
    ("joints", 1),
)

MODALITY_TEST_MODE = os.environ.get("MODALITY_TEST_MODE", "full")

VISION_INPUT_SCALE = 1.0
JOINT_INPUT_SCALE = 0.25


# -----------------------------------------------------------------------------
# RGB + joints
# -----------------------------------------------------------------------------

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_MODE = "rgb"
IMAGE_CHANNELS = 3
IMAGE_DIM = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS

APPEARANCE_HEIGHT = 64
APPEARANCE_WIDTH = 64
APPEARANCE_DIM = APPEARANCE_HEIGHT * APPEARANCE_WIDTH * IMAGE_CHANNELS

IMAGE_PCA_RANK = 48
IMAGE_BASIS_SAMPLES = 4000

JOINT_DIM = 9


# -----------------------------------------------------------------------------
# Temporal prediction
# -----------------------------------------------------------------------------

READOUT_K = (10, 30)
TEST_K = READOUT_K

DELTA_CLIP = 1.0

SMOOTH_WINDOW = 3

# -----------------------------------------------------------------------------
# V12 sparse temporal context
# -----------------------------------------------------------------------------

# Train with a variable number of observations without multiplying
# the number of training rows.
SPARSE_CONTEXT_TRAIN = False

# -----------------------------------------------------------------------------
# V13 hybrid temporal-context training
# -----------------------------------------------------------------------------

HYBRID_CONTEXT_TRAIN = True

# Same number of training rows. Each existing row receives exactly ONE
# temporal condition, deterministically:
#
#   20% history
#   20% OBS1
#   20% OBS2
#   20% OBS3
#   20% OBS5
#
HYBRID_CONTEXT_PATTERN = (
    "history",
    1,
    2,
    3,
    5,
)


# DMBN-inspired observation counts.
SPARSE_OBS_COUNTS = (1, 2, 3, 5)
SPARSE_MAX_OBS = 5

# Two temporal inputs are appended to every reservoir observation:
#   delta_t_norm : elapsed time since previous observation
#   t_norm       : absolute normalized time of current observation
SPARSE_TIME_FEATURES = True



# -----------------------------------------------------------------------------
# Reservoir
# -----------------------------------------------------------------------------

NEURON_COUNTS = (3000,)
RUNS = 1
RANDOM_SEED = 42

SPECTRAL_RADIUS = 0.95
LEAK_RATE = 0.25
INPUT_SCALE = 0.50


# -----------------------------------------------------------------------------
# Feature balancing
# -----------------------------------------------------------------------------

BLOCK_WEIGHTS = {
    "state": 1.0,
    "small": 1.0,
    "q": 2.0,
}


# -----------------------------------------------------------------------------
# Ridge
# -----------------------------------------------------------------------------

RIDGE_GRID = np.logspace(-10, 1, 12)
RIDGE_GROUP_FOLDS = 5
MIN_CLASS_ROWS = 8


# -----------------------------------------------------------------------------
# Training sensor noise
# -----------------------------------------------------------------------------

TRAIN_SENSOR_NOISE = True

TRAIN_NOISE_LEVELS = (
    0.01,
    0.025,
    0.05,
)

TRAIN_NOISE_CONDITIONS = (
    "rgb",
    "joints",
    "both",
)

TRAIN_NOISE_COPIES = 7


# -----------------------------------------------------------------------------
# Matrix-free Ridge
# -----------------------------------------------------------------------------

TRAIN_FEATURE_CHUNK = 24

RIDGE_CV_MAX_ROWS = 1200
RIDGE_CV_OUTPUTS = 64

RIDGE_CG_TOL = 3e-4
RIDGE_CG_MAX_ITER = 400

RIDGE_OUTPUT_CHUNK = 512


# -----------------------------------------------------------------------------
# Result/model tags
# -----------------------------------------------------------------------------
def k_tag():
    return "k" + "-".join(str(k) for k in READOUT_K)

def _float_tag(x):
    return f"{float(x):.6g}".replace("-", "m").replace(".", "p")


def train_split_tag():
    if TRAIN_SPLITS == ("train",):
        return "train80"

    if TRAIN_SPLITS == ("train", "validation"):
        return "train90"

    return "train_" + "_".join(TRAIN_SPLITS)


def _dropout_pattern_tag():
    short = {
        "full": "f",
        "vision": "v",
        "joints": "j",
    }

    return "-".join(
        f"{short[name]}{int(count)}"
        for name, count in MODALITY_DROPOUT_PATTERN
    )


def modality_test_tag():
    tags = {
        "full": "full",
        "vision": "vision_only",
        "joints": "joints_only",
    }

    if MODALITY_TEST_MODE not in tags:
        raise ValueError(
            "MODALITY_TEST_MODE must be full, vision or joints"
        )

    return tags[MODALITY_TEST_MODE]


def experiment_tag():
    if EXPERIMENT_MODE != "dropout":
        raise ValueError(
            "Only EXPERIMENT_MODE='dropout' is supported"
        )

    if IMAGE_HEIGHT % APPEARANCE_HEIGHT or IMAGE_WIDTH % APPEARANCE_WIDTH:
        raise ValueError(
            "Image dims must be divisible by appearance dims"
        )

    if VISION_INPUT_SCALE <= 0:
        raise ValueError(
            "VISION_INPUT_SCALE must be > 0"
        )

    if not TRAIN_NOISE_LEVELS or any(
        float(x) <= 0
        for x in TRAIN_NOISE_LEVELS
    ):
        raise ValueError(
            "TRAIN_NOISE_LEVELS must contain values > 0"
        )

    if TRAIN_NOISE_COPIES < 1:
        raise ValueError(
            "TRAIN_NOISE_COPIES must be >= 1"
        )

    levels = "-".join(
        _float_tag(x)
        for x in TRAIN_NOISE_LEVELS
    )

    return (
        f"{MODEL_NAME}_matrixfree_{IMAGE_MODE}_{IMAGE_HEIGHT}x{IMAGE_WIDTH}"
        f"_pca{IMAGE_PCA_RANK}"
        f"_{k_tag()}"
        f"_cc"
        f"_jdelta1"
        f"_vanchor1"
        f"_mdrop_{_dropout_pattern_tag()}_gcv{RIDGE_GROUP_FOLDS}"
        f"_vis{_float_tag(VISION_INPUT_SCALE)}"
        f"_{train_split_tag()}"
        f"_mf"
        f"_cv{RIDGE_CV_MAX_ROWS}"
        f"_o{RIDGE_CV_OUTPUTS}"
        f"_tol{_float_tag(RIDGE_CG_TOL)}"
        f"_it{RIDGE_CG_MAX_ITER}"
        f"_tnoise_{levels}_c{TRAIN_NOISE_COPIES}"
    )


MODEL_DIR = (
    RESULTS_DIR
    / "models"
    / experiment_tag()
)

TEST_RESULTS_DIR = (
    RESULTS_DIR
    / "testing"
    / experiment_tag()
    / modality_test_tag()
)