from src.config import Config
import torch

def test_paths():
    """
    Function to test the path of the Data Input
    """
    print("Checking Project Paths ")
    print(f"Project Root : {Config.PROJECT_ROOT}")
    print(f"Data Root : {Config.DATA_ROOT}")

    print(f"Train Data : {Config.TRAIN_DIR}")
    print(f"Test Data : {Config.TEST_DIR}")
    print(f"Val Data : {Config.VAL_DIR}")

    assert Config.TRAIN_DIR.exists(), "Train Folder is Missing !"
    assert Config.TEST_DIR.exists(), "Test Folder is Missing !"
    assert Config.VAL_DIR.exists(), "Val Folder is Missing !"


    #     DATA_ROOT = PROJECT_ROOT / "data" / "raw"

    # TRAIN_DIR = DATA_ROOT / "train"
    # VAL_DIR = DATA_ROOT / "val"
    # TEST_DIR = DATA_ROOT / "test"


if __name__ == "__main__":
    test_paths()