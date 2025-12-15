from src.config import Config
from src.modelling.model import MyResNet18
from src.data.dataset import get_dataloaders, get_train_data_loaders, get_test_data_loaders, get_val_data_loaders

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from torch import optim

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import models

cfg = Config()

def test_predict():

    model = MyResNet18(num_classes= 6)

    model.load_state_dict(torch.load(cfg.model, weights_only= True))
    model = model.to(device)
    model.eval(); # Set to inference mode

    all_preds = []
    all_labels = []

    correct = 0
    total = 0

    total_image_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            total_image_count += len(labels)

            # print(images.shape)
            # print(labels.shape)

            outputs = model(images)
            _, predicted = outputs.max(1)

            # Accuracy Tracting 
            correct += (predicted ==  labels).sum().item()
            total += labels.size(0)

            # Store for Confusion Matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Printing total image count
    print("Images covered:", total_image_count)
    print("Total test images:", len(test_dataset))

    # Final Accuracy

    test_acc = correct / total
    print(f"Test Accuracy : {100*test_acc:.2f} %")

    # Confusion matrix

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds, target_names=train_dataset.class_to_idx.keys()))

if __name__ == "__main__":
    test_predict()