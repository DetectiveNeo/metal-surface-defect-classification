from pathlib import Path


class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = PROJECT_ROOT / "data" / "raw"

    TRAIN_DIR = DATA_ROOT / "train"
    VAL_DIR = DATA_ROOT / "val"
    TEST_DIR = DATA_ROOT / "test"

    # Image / dataloader config
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # tune based on your CPU
    PIN_MEMORY = True  # good if using GPU

    # Classification related
    NUM_CLASSES = 6  # change to your actual number
