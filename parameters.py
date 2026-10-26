"""Parameters for ManiSkill multimodal direct PCA ESN v5 (class + interaction context)."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "maniskill_multimodal" / "multimodal_object_context.h5"
RESULTS_DIR = ROOT / "Results_esn_multitask"

# Mode
EXPERIMENT_MODE = "multimodal"   # "vision" or "multimodal"
USE_GOAL = False

# Interaction context from the mask: [centroid_x, centroid_y, area] per observed frame
USE_INTERACTION_CONTEXT = True
INTERACTION_MASK_KEY = "interaction_mask"
INTERACTION_INPUT_SCALE = 0.50

# Image
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_MODE = "rgb"
IMAGE_CHANNELS = 3 if IMAGE_MODE == "rgb" else 1
IMAGE_DIM = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS

APPEARANCE_HEIGHT = 64
APPEARANCE_WIDTH = 64
APPEARANCE_DIM = APPEARANCE_HEIGHT * APPEARANCE_WIDTH * IMAGE_CHANNELS

# Future RGB PCA
IMAGE_PCA_RANK = 48
IMAGE_BASIS_SAMPLES = 4000

# Proprioception / goal
JOINT_DIM = 9
GOAL_DIM = 3
JOINT_INPUT_SCALE = 0.25
GOAL_INPUT_SCALE = 0.25

# Forecast points
READOUT_K = (5, 15, 20)
TEST_K = READOUT_K

# ESN
NEURON_COUNTS = (2000,)
RUNS = 1
RANDOM_SEED = 42
SPECTRAL_RADIUS = 0.95
LEAK_RATE = 0.25
INPUT_SCALE = 0.20

# Readout
BLOCK_WEIGHTS = {
    "state": 1.0,
    "small": 1.0,
    "q": 2.0,
    "interaction": 2.0,
    "goal": 2.0,
}
RIDGE_GRID = np.logspace(-10, 1, 12)

# Pick/Push class conditioning
CLASS_CONDITIONAL = True
MIN_CLASS_ROWS = 8

# Reconstruction
DELTA_CLIP = 1.0
SMOOTH_WINDOW = 3

# Training
TRAINING_EPSILONS = (0.0,)
AUGMENTATION_COPIES = 0
AUGMENT_FLIP = False


def experiment_tag():
    if EXPERIMENT_MODE not in ("vision", "multimodal"):
        raise ValueError("EXPERIMENT_MODE must be vision or multimodal")
    if IMAGE_MODE not in ("rgb", "grayscale"):
        raise ValueError("IMAGE_MODE must be rgb or grayscale")
    if IMAGE_HEIGHT % APPEARANCE_HEIGHT or IMAGE_WIDTH % APPEARANCE_WIDTH:
        raise ValueError("Image dims must be divisible by appearance dims")
    return (
        f"{EXPERIMENT_MODE}_{IMAGE_MODE}_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_"
        f"goal{int(USE_GOAL)}_pca{IMAGE_PCA_RANK}_"
        f"{'cc' if CLASS_CONDITIONAL else 'nocc'}_"
        f"ictx{int(USE_INTERACTION_CONTEXT)}_v5"
    )


MODEL_DIR = RESULTS_DIR / "models" / experiment_tag()
TEST_RESULTS_DIR = RESULTS_DIR / "testing" / experiment_tag()