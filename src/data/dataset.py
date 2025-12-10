"""
Docstring for src.data.dataset

This module contains the class custome data loading class definition and preprocessing and 
get_loaders function to create the data loaders.

"""

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from torch.utils.data import DataLoader

from src.config import Config

cfg = Config()

class MetalSurfaceDataset(Dataset):
    """
    Custom Data Loading class for loading the images
    """
    def __init__(self, root_dir, transform=None):
        """
        root_dir -> path to the train folder
        train/ 
            Crazing/
            Inclusion/
            ...
        """

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {} # mapping class names to integers

        # Step 1 : Create class index mapping
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # Step 2 : iterate through each class folder
        for cls_name in classes:
            cls_folder = os.path.join(root_dir,cls_name)
            label = self.class_to_idx[cls_name]

            # Step 3 : Iterate thrugh each image in one class folder
            for file_name in sorted(os.listdir(cls_folder)):
                file_path = os.path.join(cls_folder, file_name)
                
                # Append image path and corresponding label
                self.image_paths.append(file_path)
                self.labels.append(label)
     
    def __len__(self):
        """ Return total count of images. """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Getting a sample image 
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms (resize, normalize, etc.)
        if self.transform:
            img = self.transform(img)

        return img, label


train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean= [0.485,0.456, 0.406],
        std= [0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = val_transform

def get_dataloaders(train_dir, val_dir, test_dir, batch_size=32, num_workers=0):
    """
    Function to get the data loaders for train, validation and test datasets
    
    :param train_dir: train directory location
    :param val_dir: validation directory location
    :param test_dir: test directory location
    :param batch_size: batch size to be kept for the data loaders
    :param num_workers: number of workers to be kept for the data loaders

    Returns :
        Train , Validation and Test Data Loaders
        And Class_Name to index mapping dictionery

    """

    train_ds = MetalSurfaceDataset(train_dir, transform=train_transform)
    val_ds   = MetalSurfaceDataset(val_dir, transform=val_transform)
    test_ds  = MetalSurfaceDataset(test_dir, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx


def get_train_data_loaders(train_dir, batch_size=32, num_workers=0):
    """
    Function to get the data loader for train dataset
    
    :param train_dir: train directory location
    :param batch_size: batch size to be kept for the data loaders
    :param num_workers: number of workers to be kept for the data loaders

    Returns :
        Train 
        And Class_Name to index mapping dictionery

    """

    train_ds = MetalSurfaceDataset(train_dir, transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, len(train_ds) , train_ds.class_to_idx


def get_val_data_loaders(val_dir, batch_size=32, num_workers=0):
    """
    Function to get the data loader for val dataset
    
    :param val_dir: val directory location
    :param batch_size: batch size to be kept for the data loaders
    :param num_workers: number of workers to be kept for the data loaders

    Returns :
        val 
        And Class_Name to index mapping dictionery

    """

    val_ds = MetalSurfaceDataset(val_dir, transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    return val_loader, len(val_ds) , val_ds.class_to_idx

def get_test_data_loaders(test_dir, batch_size=32, num_workers=0):
    """
    Function to get the data loader for test dataset
    
    :param test_dir: test directory location
    :param batch_size: batch size to be kept for the data loaders
    :param num_workers: number of workers to be kept for the data loaders

    Returns :
        test 
        And Class_Name to index mapping dictionery

    """

    test_ds = MetalSurfaceDataset(test_dir, transform=test_transform)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    return test_loader, len(test_ds) , test_ds.class_to_idx

if __name__ == "__main__":

    (train_loader, dataset_length, class_to_idx) = get_train_data_loaders(
        train_dir= cfg.data["train_dir"],
        batch_size= cfg.data["batch_size"],
        num_workers= cfg.data["num_workers"],
    )

    print(dataset_length)

    (val_loader, dataset_length, class_to_idx) = get_val_data_loaders(
    val_dir= cfg.data["val_dir"],
    batch_size= cfg.data["batch_size"],
    num_workers= cfg.data["num_workers"],
    )

    print(dataset_length)

    (test_loader, dataset_length, class_to_idx) = get_test_data_loaders(
        test_dir= cfg.data["test_dir"],
        batch_size= cfg.data["batch_size"],
        num_workers= cfg.data["num_workers"],
    )

    print(dataset_length)