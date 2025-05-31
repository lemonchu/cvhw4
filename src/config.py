# Configuration and Hyperparameters

# Dataset
IMAGE_DIR = '/home/chumeng/lab/nfs/cvhw4/gtFine/train'
MASK_DIR = '/home/chumeng/lab/nfs/cvhw4/leftImg8bit/train'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# Model
NUM_CLASSES = 19  # after mapping to Cityscapes trainIds
ENCODER = 'resnet34' # Example, from segmentation_models_pytorch
ENCODER_WEIGHTS = 'imagenet'

# Cityscapes label‐ID → trainId (0–18). All other IDs → 255 (ignore)
LABEL_ID_TO_TRAIN_ID = {
    7:  0,   # road
    8:  1,   # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}

# Training
DEVICE = 'cuda' # or 'cpu'
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NUM_FOLDS = 5  # For cross-validation

# Output
MODEL_SAVE_PATH = 'models/'
RESULTS_SAVE_PATH = 'results/'
VISUALIZATION_COUNT = 5 # Number of validation images to visualize
NUM_WORKERS = 0 # Number of workers for data loading
PIN_MEMORY = False # Whether to load the entire dataset into memory