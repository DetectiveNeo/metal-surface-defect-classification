from pathlib import Path
import os
import yaml

class Config:
    """
    Docstring for Config

    Class defined to load the config yaml file to load all the parameters that are defined in the config yaml files.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1] # Go up two folders to the project root folder

    def __init__(self, config_path: str):

        config_path = self.PROJECT_ROOT / config_path

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        data_cfg = cfg["data"]

        self.train_dir = data_cfg["train_dir"]
        self.val_dir = data_cfg["val_dir"]
        self.test_dir = data_cfg["test_dir"]
        
        self.batch_size = data_cfg["batch_size"]
        self.num_workers = data_cfg["num_workers"]

        self.num_of_classes = data_cfg["num_of_classes"]

