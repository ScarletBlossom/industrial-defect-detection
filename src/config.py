import torch

class Config:
    # Paths
    DATA_DIR = "data"
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/val"
    TEST_DIR = "data/test"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # Model
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 2
    PRETRAINED = True
    
    # Training
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Data
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data generation
    NUM_TRAIN_SAMPLES = 1000
    NUM_VAL_SAMPLES = 200
    NUM_TEST_SAMPLES = 100
    
    # Random seed
    SEED = 42