import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet34_Weights
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

from Application import custom_collate_fn
from dataLoaders.OXFordData import OXFordData


# 假设 PetsNet 类定义已经给出
class PetsNet(nn.Module):
    def __init__(self, num_classes=37):
        super(PetsNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()    # 移除原有的全连接
        self.classifier_fc = nn.Linear(in_features, self.num_classes)
        self.bound_fc = nn.Linear(in_features, 4)

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier_fc(features)
        bbox = self.bound_fc(features)
        return bbox, class_logits

# 加载模型和权重
model = PetsNet(37)
model.load_state_dict(torch.load('/runs/2/best.pt'))
model.eval()  # 将模型设置为评估模式

# 假设 valid_loader 已经定义
# 初始化列表来收集预测和真实标签
true_labels = []
pred_labels = []

train_val_set = OXFordData('/home/wsx/ptJobs/TB20201576A5/data/', mode='train')
total_size = len(train_val_set)
val_size = int(total_size * 0.05)
train_size = total_size - val_size

train_set, valid_set = random_split(train_val_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=12, collate_fn=custom_collate_fn)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=12, collate_fn=custom_collate_fn)


# 禁用梯度计算
with torch.no_grad():
    for inputs, labels_bd, labels in valid_loader:
        # 假设labels是分类的真实标签
        _, logits = model(inputs)
        predictions = torch.argmax(logits, dim=1)
        true_labels.extend(labels.numpy())
        pred_labels.extend(predictions.numpy())

# 计算混淆矩阵
cm = confusion_matrix(true_labels, pred_labels)

# 绘制混淆矩阵
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap='OrRd')
plt.ylabel('Actual labels')
plt.xlabel('Predicted labels')
plt.show()
