from src.config import Config

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MyResNet18(nn.Module):
    def __init__(self, num_classes= 6, freeze_backbone= True):
        super().__init__()

        # Load pretrained Resnet18
        self.model = models.resnet18(weights= models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze the backbone if needed
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    
    def forward(self, x):
        return self.model(x)


cfg = Config()


if __name__ == "__main__":

    model = MyResNet18(num_classes= 6, freeze_backbone= True)

    print(model)