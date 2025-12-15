from pathlib import Path
import yaml

class Config:
    """
    Docstring for Config

    Class defined to load the config yaml file to load all the parameters that are defined in the config yaml files.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1] # Go up two folders to the project root folder

    def __init__(self):

        config_path = self.PROJECT_ROOT / "configs/config.yaml"

        self.model_save_path = self.PROJECT_ROOT / "models/resnet18_best.pth"
        
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            self.cfg = cfg
            self.data = cfg["data"]
            self.model = cfg["model"]
            self.train = cfg["train"]
           
if __name__ == "__main__":
    cfg = Config()

    # cfg.PROJECT_ROOT + "models/resnet18_best.pth"

    # print(cfg.PROJECT_ROOT / "models/resnet18_best.pth")

    print(cfg.model_save_path)



