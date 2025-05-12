import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
from nets.resnet50 import Bottleneck, ResNet
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import os
import yaml
from pathlib import Path
import imgaug.augmenters as iaa

# 假设 config 文件夹下有 config.yml
config_path = Path(__file__).parent / "config" / "config_train.yaml"

# 读取 YAML 文件
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


# 数据增强定义
seq = iaa.Sequential([
    #iaa.Crop(px=(1, 16), keep_size=False),    # 随机裁剪
    iaa.Fliplr(0.5),                          # 50%几率水平翻转
    iaa.Sometimes(0.5, iaa.Affine(rotate=180)),  # 50% 概率旋转180度（上下颠倒）
    iaa.GaussianBlur(sigma=(0, 3.0))          # 高斯模糊
])

# 定义数据转换
transform_train = transforms.Compose([
    #transforms.Resize((384, 384)),  # 调整图像大小  384 224
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((500, 500)),  # 调整图像大小  384 224
    transforms.ToTensor(),
])

# 自定义数据集类，结合 imgaug 进行增强
class ImgAugmentedDataset(Dataset):
    def __init__(self, root_dir, transform=None, augmentor=None):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentor = augmentor

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # 遍历目录结构，加载所有图片
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx

            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(class_path, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图像路径和标签
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载为 RGB 图像
        image = Image.open(image_path).convert("RGB")

        # 转为 numpy 数组 (H, W, C)
        image_np = np.array(image)

        # 应用数据增强（imgaug 通常操作 numpy）
        if self.augmentor:
            image_np = self.augmentor(image=image_np)  # 仍是 (H, W, C)

        # 使用 cv2 或 PIL 做 resize（更稳）
        image_np = cv2.resize(image_np, (500, 500))  # (H, W, C)

        # 转回 PIL
        image = Image.fromarray(image_np.astype(np.uint8))  # Safe to convert

        # 应用 transform（注意：只调用一次）
        if self.transform:
            image = self.transform(image)

        return image, label


# 训练和验证函数优化
def fit_one_epoch(net, softmaxloss, epoch, epoch_size, epoch_size_val, gen, gen_test, Epoch, cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    print('\nStart train')
    with tqdm(total=epoch_size, desc='Epoch{}/{}'.format(epoch + 1, Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            images, targets = batch[0], batch[1]
            if cuda:
                images = images.cuda()
                targets = targets.cuda()

            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = net(images)
            
            # 计算损失
            loss = softmaxloss(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr':  optimizer.param_groups[0]['lr']})
            pbar.update(1)
    torch.cuda.empty_cache()       # 清理未使用的缓存显存（可选）
    torch.cuda.ipc_collect()       # 清理跨进程共享内存（通常在多进程下有用）
    torch.cuda.synchronize()       # 等待 CUDA 操作完成（可选）

    # 测试阶段
    net.eval()
    print('\nStart test')
    test_correct = 0
    with tqdm(total=epoch_size_val, desc='Epoch{}/{}'.format(epoch + 1, Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_test):
            images, targets = batch[0], batch[1]
            if cuda:
              images = images.cuda()
              targets = targets.cuda()
            
            outputs = net(images)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == targets.data)
            pbar.set_postfix(**{'test AP': float(100 * test_correct / len(test_dataset))})
            pbar.update(1)
    torch.cuda.empty_cache()       # 清理未使用的缓存显存（可选）
    torch.cuda.ipc_collect()       # 清理跨进程共享内存（通常在多进程下有用）
    torch.cuda.synchronize()       # 等待 CUDA 操作完成（可选）
    #torch.save(net.state_dict(), 'logs/Epoch{}-Total_Loss{}.pth'.format((epoch + 1), (total_loss / ((iteration + 1)))))


if __name__ == '__main__':
    cuda = config["cuda"]
    pre_train = config["pre_train"]
    CosineLR = config["CosineLR"]

    lr = float(config["lr"])
    Batch_size = config["Batch_size"]
    Init_Epoch = config["Init_Epoch"]
    Fin_Epoch = config["Fin_Epoch"]

    # 创建模型
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=config["num_classes"])
    if pre_train:
        model_path = config["pre_train_path"]
        model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_dataset = ImgAugmentedDataset(root_dir=config["train_dataset_path"], transform=transform_train, augmentor=seq)
    test_dataset = ImageFolder(root=config["test_dataset_path"], transform=transform_test)

    gen = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True, num_workers=config["num_workers"], pin_memory=True)
    gen_test = DataLoader(dataset=test_dataset, batch_size=Batch_size//2, shuffle=True, num_workers=config["num_workers"], pin_memory=True)

    epoch_size = len(gen)
    epoch_size_val = len(gen_test)

    softmax_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if CosineLR:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-10)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    for epoch in range(Init_Epoch, Fin_Epoch):
        fit_one_epoch(net=model, softmaxloss=softmax_loss, epoch=epoch, epoch_size=epoch_size,
                      epoch_size_val=epoch_size_val, gen=gen, gen_test=gen_test, Epoch=Fin_Epoch, cuda=cuda)
        lr_scheduler.step()
        if epoch == 49 or epoch == 99:  
            torch.save(model.state_dict(), f"logs/model_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), 'logs/final_model.pth')
