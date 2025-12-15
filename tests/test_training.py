# from src.models import model
from src.modelling.train import train_one_epoch, val_one_epoch
from src.config import Config
from src.modelling.model import MyResNet18

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()

def test_train_one_epoch():
    """
    test one poch of the train function 
    """
    model = MyResNet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr= cfg.train['lr'])
    criterion = nn.CrossEntropyLoss()

    # create a dummy dataloader
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 6, (4,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)


    (train_epoch_loss, train_epoch_acc) = train_one_epoch(
        model= model,
        train_loader= loader,
        optimizer= optimizer,
        criterion= criterion,
        train_dataset_length= len(dataset),
        epoch= 1,
        num_epochs= 1,
        device= device,
    )

    assert isinstance(train_epoch_loss, float) , "Train epoch Loss is not a float"

    assert isinstance(train_epoch_acc, float) , "Train epoch Acc is not a float"

def test_valid_one_epoch():
    """
    test one poch of the valid function 
    """
    model = MyResNet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr= cfg.train['lr'])
    criterion = nn.CrossEntropyLoss()

    # create a dummy dataloader
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 6, (4,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)


    (valid_epoch_loss, valid_epoch_acc) = val_one_epoch(
        model= model,
        val_loader= loader,
        optimizer= optimizer,
        criterion= criterion,
        val_dataset_length= len(dataset),
        epoch= 1,
        num_epochs= 1,
        device= device,
    )

    assert isinstance(valid_epoch_loss, float) , "valid epoch Loss is not a float"

    assert isinstance(valid_epoch_acc, float) , "valid epoch Acc is not a float"

# test_val_one_epoch.py

# same as above but with val loader

# ensure eval mode used

# test_checkpoint.py

# create dummy model

# call save_checkpoint

# assert file exists

# test_train_function.py

# override cfg.train["epochs"] = 1

# override cfg.train["patience_level"] = 1

# call train(cfg)

# assert completes without crash


if __name__ == "__main__":
    test_train_one_epoch();
    test_valid_one_epoch();