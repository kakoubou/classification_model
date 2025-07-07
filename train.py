import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from nets.resnet50 import Bottleneck, ResNet
from utils import seq, ImgAugmentedDataset, fit_one_epoch

if __name__ == '__main__':
    config_path = Path(__file__).parent / "config" / "config_train.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    cuda = config["cuda"]
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=config["num_classes"])
    if config["pre_train"]:
        model.load_state_dict(torch.load(config["pre_train_path"]))
    model = model.to(device)

    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Resize((500, 500)), transforms.ToTensor()])

    train_dataset = ImgAugmentedDataset(config["train_dataset_path"], transform=transform_train, augmentor=seq)
    val_dataset = datasets.ImageFolder(config["val_dataset_path"], transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=config["Batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["Batch_size"] // 2, shuffle=False, num_workers=config["num_workers"], pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-10)
                 if config["CosineLR"]
                 else torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92))

    for epoch in range(config["Init_Epoch"], config["Fin_Epoch"]):
        fit_one_epoch(model, loss_fn, epoch, len(train_loader), len(test_loader),
                      train_loader, val_loader, config["Fin_Epoch"], cuda,
                      optimizer, len(val_dataset))

        scheduler.step()

        if epoch in [49, 99]:
            torch.save(model.state_dict(), f"logs/model_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), 'logs/final_model.pth')

