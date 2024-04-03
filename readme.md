# World is a ball 深度学习代码框架

本工程旨在降低进行分类、识别（定位+分类）、回归等任务时的工作难度。使研究生和开发人员集中精力在项目的核心部分，而不是花时间写重复的无意义重复代码。clone到本地后，按照实际需求进行调整的部分有：

- dataloader文件夹下创建自己的数据加载

- model文件夹下创建自己的模型类

- Application中进行微调（可选）

下面将通过几个具体的例子来演示该项目的用法



## 1. 目标识别（定位+分类）以宠物分类数据集[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)  为例

### 1.1 数据加载dataloader

数据加载函数需要继承自**torch.utils.data.Dataset**类，根据实际情况实现**\_\_init()\_\_**、**\_\_len()\_\_** 、**\_\_getitem()\_\_** 三个方法即可。主要是 **\_\_getitem\_\_()** 方法，返回数据和标签。由于本例实现的是目标检测，这里返回三个要素：数据、边界、类别。

```python
class OXFordData(Dataset):
    def __init__(self, data_path, mode='train'):
        self.data_path = data_path
        self.image_path = os.path.join(self.data_path, 'images')
        self.xml_path = os.path.join(data_path, 'annotations/xmls')
        if mode == 'train':
            self.txt_file = data_path + '/annotations/trainval.txt'
        elif mode == 'test':
            self.txt_file = data_path + '/annotations/test.txt'
        self.data_lines = self.get_list()
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def get_list(self):
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
        return lines

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        try:
            line = self.data_lines[index].strip()
            base_name = line.split(' ')[0]
            file_name, label_class = f"{base_name}.jpg", line.split(' ')[1]
            label_class = torch.tensor(int(label_class) - 1, dtype=torch.long)

            img_path = os.path.join(self.image_path, file_name)
            image = Image.open(img_path).convert('RGB')

            if self.transforms:
                image = self.transforms(image)

            xml_file = os.path.join(self.xml_path, f"{base_name}.xml")
            label_bd = torch.tensor(parse_xml_file(xml_file), dtype=torch.float)
        except Exception as e:
            # 如果你希望在出错时看到错误消息，可以取消下面这行的注释
            # print(f"Error loading data: {e}")
            return None, None, None

        return image, label_bd, label_class
```

返回值可以是列表、元祖或者字典，不同的数据结构需要再Applications.py中做出对应的修改。



### 1.2 网络设计

数据加载完成后，设计网络模型，网络模型应继承自torch.nn.Module类。为完成本例中的任务，修改resnet50的头部，修改为定位头和分类头：

```python

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

```

可以对网络进行测试，确保处理后的图片数据经过forward可以得到正确的shape, 如：

```python
net = PetsNet(37)
data = torch.randn(32, 3, 256, 256)
class_logits, bbox = net(data)
print(class_logits.shape, bbox.shape)
```



### 1.3 训练代码微调

在Application.py中进行代码的微调，由于当前版本就是基于本例做的，这部分可以不做，直接修改main.py中的超参数，开始训练





训练得到的参数变化过程

![](https://s2.loli.net/2024/04/02/jCfTxREAln3v8cI.png)

验证集的混淆矩阵

![](https://s2.loli.net/2024/04/02/BLXksSrDyZdjzba.png)

定位和识别结果

![](https://s2.loli.net/2024/04/02/W5kPK7SCNDHfGj8.png)
