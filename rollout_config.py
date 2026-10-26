"""Configuration for ManiSkill physical ESN rollout diagnostics.

This file is intentionally separate from parameters.py:
- parameters.py controls the trained model/experiment.
- rollout_config.py controls simulator diagnostics and visualization.
"""

# Which Python module contains load_model() and predict_trajectory().
# Current project uses the new latent-CNN ESN in esn_test.py.
ESN_MODULE = "esn_test"

DEFAULT_TASK = "pick"
DEFAULT_TRAJECTORY = 6
DEFAULT_K = 5
DEFAULT_MODE = "compare"      # oracle | esn | compare
DEFAULT_NEURONS = 1000
DEFAULT_RUN = 1
DEFAULT_SPLIT = "validation"

SIM_BACKEND = "physx_cpu"
OBS_MODE = "rgb"
CONTROL_MODE = "pd_joint_pos"

# Leave enough time for the full continuation.
MIN_TIME_LIMIT = 200
TIME_LIMIT_MARGIN = 10

# Video / diagnostic outputs.
OUTPUT_DIR = "sim_rollouts_diagnostic"
SAVE_VIDEO_BY_DEFAULT = False
GIF_SIZE = 384
GIF_DURATION_MS = 120

# Diagnostic thresholds only; these do NOT define ManiSkill success.
#
# In the Panda demos each finger qpos is approximately:
#   open   ~ 0.040 m
#   closed ~ 0.018 m
# We use the mean of qpos[7:9].
GRIPPER_CLOSING_THRESHOLD = 0.025

# "Near object" is also only diagnostic. It helps mark the critical
# pre-grasp interval in plots/CSV.
NEAR_OBJECT_DISTANCE_M = 0.030

# Plot a zoomed diagnostic around the first predicted closing event.
PRE_CLOSE_STEPS = 12
POST_CLOSE_STEPS = 12

# Save one CSV per rollout and summary plots.
SAVE_CSV = True
SAVE_PLOTS = True

# Numerical display precision.
PRINT_PRECISION = 4
