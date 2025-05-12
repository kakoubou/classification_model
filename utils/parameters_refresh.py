from nets.resnet50 import Bottleneck, ResNet
import torch
import yaml
from pathlib import Path


# 假设 config 文件夹下有 config.yml
config_path = Path(__file__).parent.parent / "config" / "config_train.yaml"

# 读取 YAML 文件
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


# 定义模型（修改输入通道为3）
model = ResNet(Bottleneck, [3,4,6,3], num_classes=config["num_classes"])

# 加载预训练权重
pretrained = torch.load("logs/resnet50_pretrain_download.pth")
model_dict = model.state_dict()

# 筛选参数
pretrained = {k: v for k, v in pretrained.items() 
             if k in model_dict and v.shape == model_dict[k].shape}

# 更新参数
model_dict.update(pretrained)
model.load_state_dict(model_dict)

# 冻结参数
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

# 保存新权重
torch.save(model.state_dict(), "logs/resnet50_pretrain.pth")