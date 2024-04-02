import os

import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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


def parse_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 解析图像尺寸
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # 解析对象的边界框信息，并转换为比例
    obj = root.find('object')
    bnd_box = obj.find('bndbox')
    x_min = round(int(bnd_box.find('xmin').text) / width, 4)
    y_min = round(int(bnd_box.find('ymin').text) / height, 4)
    x_max = round(int(bnd_box.find('xmax').text) / width, 4)
    y_max = round(int(bnd_box.find('ymax').text) / height, 4)

    return [x_min, y_min, x_max, y_max]


# def show_image_tensor(image_tensor):
#     # 将张量转换回PIL图像，这里使用`ToPILImage`转换
#     to_pil = transforms.ToPILImage()
#     image = to_pil(image_tensor)
#
#     # 使用matplotlib进行显示
#     plt.imshow(image)
#     plt.axis('off')  # 不显示坐标轴
#     plt.show()
#
#
if __name__ == '__main__':
    test_data = OXFordData('/home/wsx/ptJobs/TB20201576A5/data/', mode='train')
    for i in range(len(test_data)):
        if test_data.__getitem__(i)[0] is not None:
            # print(int(test_data.__getitem__(i)[2]))
            pass
