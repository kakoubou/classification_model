# test.py

import torch
from nets.resnet50 import ResNet, Bottleneck
from torchvision import transforms
import yaml
from pathlib import Path

from utils import test_model

# Load configuration file
config_path = Path(__file__).parent / "config" / "config_train.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Define image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
])

# Load the model
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=config["num_classes"])
checkpoint_path = 'logs/model_epoch100.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

# Evaluate the model, generate and save confusion matrix and accuracy
test_model(
    model=model,
    test_dataset_path=config["test_dataset_path"],
    transform=transform,
    batch_size=128,
    save_dir="logs"
)


