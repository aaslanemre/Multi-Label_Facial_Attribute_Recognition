import torch

# --- 1. Experiment Setup ---
MLFLOW_EXPERIMENT_NAME = "CelebA_Attribute_Comparison"
MLFLOW_RUN_NAME = "ViT_Frozen_Transfer_Run_1" # CHANGE ME fViT_Frozen_Transfer_Run_1"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Data Setup ---
HF_DATASET_NAME = "flwrlabs/celeba"
NUM_CLASSES = 40 # Total number of facial attributes (FACES in the dataset)
# The batch size for validation and testing should be higher for efficiency
BATCH_SIZE_EVAL = 128 

# --- 3. Model and Training Hyperparameters ---
MODEL_NAME = "ViT" # Options: "ResNet50", "ViT"

NUM_EPOCHS = 20
BATCH_SIZE_TRAIN = 64
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5

# --- 4. Transfer Learning Configuration ---
# Set to True to freeze all convolutional/transformer blocks and only train the final FC head.
FREEZE_BASE_LAYERS = False