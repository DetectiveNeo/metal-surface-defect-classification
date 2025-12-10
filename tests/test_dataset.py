import torch
from src.config import Config
from src.data.dataset import get_dataloaders

def test_loading():
    """
    Docstring for test_loading

    test loading of the data done my get_loaders function

    """

    cfg = Config("configs/data.yaml")

    (train_loader, val_loader, test_loader, class_to_idx) = get_dataloaders(
        train_dir= cfg.train_dir,
        val_dir= cfg.val_dir,
        test_dir= cfg.test_dir,
        batch_size= cfg.batch_size,
        num_workers= cfg.num_workers
    )

    assert isinstance(class_to_idx, dict) , "Class_to_idx is not a dictionery"

    assert len(class_to_idx) == cfg.num_of_classes, "Number of Classes do not Match"

    images, labels = next(iter(train_loader))

    assert isinstance(images, torch.Tensor) , "Images variable is not a torch tensor"
    assert isinstance(labels, torch.Tensor) , "labels variable is not a torch tensor"

    assert images.shape[0] == cfg.batch_size , "Number of images in the batch is not equal to the batch_size"
    assert images.ndim == 4 , "Images batch instance ndim is not equal to 4"

    assert labels.shape[0] == cfg.batch_size , "Number of labels in the batch is not equal to the batch_size"

# if __name__ == "__main__":
    
#     print("Loading Config..")
#     cfg = Config("configs/data.yaml")

#     print("Creating dataloaders...")
#     (train_loader, val_loader, test_loader, class_to_idx) = get_dataloaders(
#         train_dir= cfg.train_dir,
#         val_dir= cfg.val_dir,
#         test_dir= cfg.test_dir,
#         batch_size= cfg.batch_size,
#         num_workers= cfg.num_workers
#     )

#     print(f"Classes Detected : {class_to_idx}")

#     # Test one batch
#     print("Fetch one batch from Train Loader..")
#     images, labels = next(iter(train_loader))

#     print(f"Image batch shape : {images.shape}")
#     print(f"Label batch shape : {labels.shape}")

#     print("Dataset pipeline is working correctly")

