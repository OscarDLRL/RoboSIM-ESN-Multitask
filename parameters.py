"""Experiments with the official ManiSkill PickCube/PushCube demonstration dataset."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'datasets' / 'maniskill_multimodal' / 'multimodal_original.h5'
RESULTS_DIR = ROOT / 'Results_esn_multitask'

# Change these two switches for the ablation study.
EXPERIMENT_MODE = 'multimodal'  # 'vision' or 'multimodal'
USE_GOAL = False               # False: sensory-only; True: additionally condition on XYZ goal

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_MODE = 'rgb'              # 'rgb' or 'grayscale'
IMAGE_CHANNELS = 3 if IMAGE_MODE == 'rgb' else 1
IMAGE_DIM = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
APPEARANCE_HEIGHT = 16
APPEARANCE_WIDTH = 16
APPEARANCE_DIM = APPEARANCE_HEIGHT * APPEARANCE_WIDTH * IMAGE_CHANNELS
JOINT_DIM = 9
GOAL_DIM = 3

READOUT_K = (5, 15, 20, 35, 40)
TEST_K = READOUT_K
NEURON_COUNTS = (1000,)
RUNS = 1                       # Increase after the first successful validation run
RANDOM_SEED = 42
SPECTRAL_RADIUS = .95
LEAK_RATE = .25
INPUT_SCALE = .2
READOUT_RIDGE = 1e-2
DELTA_CLIP = 1.0              # Clipping only RGB difference predictions

# Extra multiplier on normalized proprioception relative to RGB pixel differences.
JOINT_INPUT_SCALE = 0.25
GOAL_INPUT_SCALE = 0.25

TRAINING_EPSILONS = (0.0,)
AUGMENTATION_COPIES = 0        # Initial version forbids augmentation; spatial flips change robot semantics
AUGMENT_FLIP = False

def experiment_tag():
    if EXPERIMENT_MODE not in ('vision', 'multimodal'):
        raise ValueError('EXPERIMENT_MODE must be vision or multimodal')
    if IMAGE_MODE not in ('rgb','grayscale'):
        raise ValueError('IMAGE_MODE must be rgb or grayscale')
    if IMAGE_HEIGHT % APPEARANCE_HEIGHT or IMAGE_WIDTH % APPEARANCE_WIDTH:
        raise ValueError('Image dims must be divisible by appearance dims')
    return f'{EXPERIMENT_MODE}_{IMAGE_MODE}_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_goal{int(USE_GOAL)}'

MODEL_DIR = RESULTS_DIR / 'models' / experiment_tag()
TEST_RESULTS_DIR = RESULTS_DIR / 'testing' / experiment_tag()

FUTURE_JOINT_SCALE = 1.0
ARCHITECTURE_TAG = 'oracle_future_q_rgb'