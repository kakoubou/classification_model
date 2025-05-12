import torch
from nets.resnet50 import ResNet,Bottleneck
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path

# 假设 config 文件夹下有 config.yml
config_path = Path(__file__).parent / "config" / "config_train.yaml"

# 读取 YAML 文件
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((500, 500)),  # 调整图像大小
    transforms.ToTensor(),
])

all_preds = []
all_labels = []
Batch_size = 128

#root = '.\logs'
#file_dir = os.listdir(root)

model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=config["num_classes"])

#for file in file_dir:
#PATH = os.path.join(root, file)

PATH = 'logs/model_epoch100.pth'

model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
model = model.cpu()
model.eval()

test_dataset = ImageFolder(root=config["test_dataset_path"], transform=transform)

gen_test = DataLoader(dataset=test_dataset, batch_size=Batch_size, shuffle=False)

test_correct = 0

with torch.no_grad():
    for inputs, labels in gen_test:
        inputs = inputs.cpu()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
acc = accuracy_score(all_labels, all_preds)

# 画图
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix\nAccuracy: {acc:.2%}')

# 保存图片
plt.savefig('logs/confusion_matrix.png')
plt.close()

# 可选：保存 accuracy 到一个 txt 文件
with open('logs/test_accuracy.txt', 'w') as f:
    f.write(f'Accuracy: {acc:.4f} ({acc:.2%})')
