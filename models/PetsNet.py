import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights


class PetsNet(nn.Module):
    def __init__(self, num_classes=37):
        super(PetsNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()    # 移除原有的全连接
        self.classifier_fc = nn.Linear(in_features, self.num_classes)
        self.bound_fc = nn.Linear(in_features, 4)

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier_fc(features)
        bbox = self.bound_fc(features)

        return bbox, class_logits


# if __name__ == '__main__':
#     net = PetsNet(37)
#     data = torch.randn(32, 3, 256, 256)
#     class_logits, bbox = net(data)
#     print(class_logits.shape, bbox.shape)
