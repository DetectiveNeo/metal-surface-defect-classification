from torch import nn
import torch
# from torchsummary import summary
from src.config import Config
from src.data.dataset import get_dataloaders
from src.models.model import MyResNet18

cfg = Config("configs/data.yaml")
model = MyResNet18(num_classes= cfg.num_of_classes, freeze_backbone= True)

def test_model_instantiation():
    """
    Test Model instantiation
    """    
    assert isinstance(model, nn.Module), "Model is not an nn.Module instance"

def test_forward_pass_shape():
    """
    Test forward pass of the model instantiated
    """

    x = torch.randn(4,3,224,224)
    out = model(x)

    assert out.shape == (4,cfg.num_of_classes) , "Model ouput is not correct {out.shape} is comming instead of (4,{cfg.num_of_classes}})"

def test_trainable_parameter_count():
    """
    Test the number of trainable parameters
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FC layer params = in_features * num_classes + bias
    expected = model.model.fc.in_features * cfg.num_of_classes + cfg.num_of_classes

    assert trainable_params == expected, "Trainable parameter count mismatch."

