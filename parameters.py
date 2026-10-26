"""Parameters for ManiSkill multimodal direct PCA ESN v3 (class-conditional)."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "maniskill_multimodal" / "multimodal_object_context.h5"
RESULTS_DIR = ROOT / "Results_esn_multitask"

# Modo
EXPERIMENT_MODE = "multimodal"   # "vision" or "multimodal"
USE_GOAL = False

# Imagen
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_MODE = "rgb"               # "rgb" or "grayscale"
IMAGE_CHANNELS = 3 if IMAGE_MODE == "rgb" else 1
IMAGE_DIM = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS

# Ultimo frame observado usado como feature (dims deben dividir a las de la imagen)
APPEARANCE_HEIGHT = 32
APPEARANCE_WIDTH = 32
APPEARANCE_DIM = APPEARANCE_HEIGHT * APPEARANCE_WIDTH * IMAGE_CHANNELS

# PCA de los deltas RGB futuros
IMAGE_PCA_RANK = 48
IMAGE_BASIS_SAMPLES = 4000

# Propiocepcion / goal
JOINT_DIM = 9
GOAL_DIM = 3
JOINT_INPUT_SCALE = 0.25
GOAL_INPUT_SCALE = 0.25

# Puntos de prediccion
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
BLOCK_WEIGHTS = {"state": 1.0, "small": 1.0, "q": 2.0, "goal": 2.0}
RIDGE_GRID = np.logspace(-10, 1, 12)   # relativo a la traza media del Gram

# Memoria condicionada por clase (Pick/Push)
CLASS_CONDITIONAL = True
MIN_CLASS_ROWS = 8

# Reconstruccion
DELTA_CLIP = 1.0
SMOOTH_WINDOW = 3

# Entrenamiento
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
        f"{'cc' if CLASS_CONDITIONAL else 'nocc'}_v3"
    )


MODEL_DIR = RESULTS_DIR / "models" / experiment_tag()
TEST_RESULTS_DIR = RESULTS_DIR / "testing" / experiment_tag()