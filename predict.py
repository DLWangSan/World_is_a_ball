import json

import torch
from PIL import Image
from matplotlib import pyplot as plt, patches

from torchvision import transforms
from models.PetsNet import PetsNet

if __name__ == '__main__':

    image_path = "/home/wsx/ptJobs/TB20201576A5/data/images/Abyssinian_60.jpg"
    class_file = "/home/wsx/ptJobs/TB20201576A5/data/class_dict.json"
    with open(class_file, 'r')as f:
        class_dict = json.load(f)
    model = PetsNet(37)
    model.load_state_dict(torch.load('/runs/2/best.pt'))
    model.eval()  # 将模型设置为评估模式

    transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image_ = transforms(image).unsqueeze(0)
    with torch.no_grad():
        label_bd, label_cls = model(image_)
        top_p, top_class = torch.topk(torch.softmax(label_cls, dim=1), 3)

    bbox = label_bd.squeeze().tolist()
    x_min, y_min, x_max, y_max = bbox
    x_min *= original_width
    x_max *= original_width
    y_min *= original_height
    y_max *= original_height

    # 创建图和轴
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 创建一个矩形patch
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')

    # 将矩形添加到轴上
    ax.add_patch(rect)

    plt.show()

    result = {class_dict[str(list(top_class[0].numpy())[i])]: list(top_p[0].numpy())[i] for i in range(len(list(top_class[0].numpy())))}
    print(result)
