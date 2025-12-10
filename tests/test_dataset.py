import torch
from src.config import Config
from src.data.dataset import get_dataloaders

def test_loading():
    """
    Docstring for test_loading

    test loading of the data done my get_loaders function

    """

    cfg = Config()

    (train_loader, val_loader, test_loader, class_to_idx) = get_dataloaders(
        train_dir= cfg.data["train_dir"],
        val_dir= cfg.data["val_dir"],
        test_dir= cfg.data["test_dir"],
        batch_size= cfg.data["batch_size"],
        num_workers= cfg.data["num_workers"]
    )

    assert isinstance(class_to_idx, dict) , "Class_to_idx is not a dictionery"

    assert len(class_to_idx) == cfg.data["num_of_classes"], "Number of Classes do not Match"

    images, labels = next(iter(train_loader))

    assert isinstance(images, torch.Tensor) , "Images variable is not a torch tensor"
    assert isinstance(labels, torch.Tensor) , "labels variable is not a torch tensor"

    assert images.shape[0] == cfg.data["batch_size"] , "Number of images in the batch is not equal to the batch_size"
    assert images.ndim == 4 , "Images batch instance ndim is not equal to 4"

    assert labels.shape[0] == cfg.data["batch_size"] , "Number of labels in the batch is not equal to the batch_size"

if __name__ == "__main__":

    cfg = Config()

    print(cfg.cfg.keys())

    test_loading()
    

