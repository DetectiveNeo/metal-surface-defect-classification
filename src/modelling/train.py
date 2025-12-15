from src.config import Config
from src.modelling.model import MyResNet18
from src.data.dataset import get_dataloaders, get_train_data_loaders, get_test_data_loaders, get_val_data_loaders

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from torch import optim

cfg = Config()

# Helper : Device selection
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import numpy as np
import torch

# Seed Fucntion
def set_seed(seed: int = 42) -> None:
    """
    Sets random seed for reproducibility across:
    - Python
    - NumPy
    - PyTorch (CPU & CUDA)

    Note:
    Full determinism can reduce performance.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Train One Epoch 
def train_one_epoch(model, train_loader, train_dataset_length,epoch, num_epochs, device, optimizer, criterion):
    """
    Train One Epoch on the training dataset
    """
    # --------------------
    # TRAINING
    # --------------------

    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_epoch_loss = train_loss / train_dataset_length
    train_epoch_acc  = train_correct / train_total

    return train_epoch_loss, train_epoch_acc

# Validate One Epoch
def val_one_epoch(model, val_loader, val_dataset_length,epoch, num_epochs, device, optimizer, criterion):
    """
    Validate One Epoch
    """
    # --------------------
    # VALIDATION
    # --------------------
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_epoch_loss = val_loss / val_dataset_length
    val_epoch_acc  = val_correct / val_total

    return val_epoch_loss, val_epoch_acc

def train(num_epochs, lr, patience_level):


    num_epochs = num_epochs

    best_val_acc = 0
    best_val_loss = float('inf')
    patience = patience_level
    patience_counter = 0

    train_losses_list = []
    val_losses_list = []
    train_accs_list = []
    val_accs_list = []

    device = get_device()
    set_seed()

    model = MyResNet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):

        # --------------------
        # TRAINING
        # --------------------

        (train_loader, train_dataset_length , _) = get_train_data_loaders(cfg.data['train_dir'], batch_size= cfg.data['batch_size'], num_workers= cfg.data['num_workers'])

        train_epoch_loss, train_epoch_acc = train_one_epoch(
            model= model,
            train_loader= train_loader,
            train_dataset_length= train_dataset_length, 
            epoch= epoch,
            num_epochs = num_epochs,
            device= device,
            optimizer= optimizer,
            criterion= criterion
        )
        
        # --------------------
        # VALIDATION
        # --------------------
        (val_loader, val_dataset_length, _) = get_val_data_loaders(cfg.data['val_dir'], batch_size= cfg.data['batch_size'], num_workers= cfg.data['num_workers'])
        
        (val_epoch_loss, val_epoch_acc) = val_one_epoch(model, val_loader, val_dataset_length,epoch, num_epochs, device, optimizer, criterion)


        # Save logs
        train_losses_list.append(train_epoch_loss)
        val_losses_list.append(val_epoch_loss)
        train_accs_list.append(train_epoch_acc)
        val_accs_list.append(val_epoch_acc)

        print("----------------------------------------------------------------------------------------------------")
        # Print Summary
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")


        # --------------------
        # CHECKPOINT (based on val accuracy)
        # --------------------
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), cfg.model_save_path)
            print("✔ Saved Best Model")

        # --------------------
        # EARLY STOPPING (based on val loss)
        # --------------------
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in val_loss for {patience_counter} epochs.")

            if patience_counter >= patience:
                print("🔥 Early stopping triggered!")
                break

        scheduler.step()

        print("----------------------------------------------------------------------------------------------------")


# Main Running pipeline
if __name__ == "__main__":

    print(get_device())

    train(
        num_epochs= cfg.train['epochs'],
        lr= cfg.train['lr'],
        patience_level= cfg.train['patience_level']
    )





