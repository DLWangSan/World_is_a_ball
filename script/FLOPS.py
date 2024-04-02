#_*_ coding:utf-8 _*_ 

import os
import torch
from thop import profile
import sys

from models.PetsNet import PetsNet

sys.path.append("../")
from torchvision.models import resnet50
from torchvision.models import mobilenet_v2


def FLOPS(model_path, name):
    model = PetsNet(37)
    model.load_state_dict(torch.load(model_path))
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, inputs=(input, ))
    print("----------------")
    print(name)
    print("flops:",flops)
    print("params:",params)
    print("----------------")

if __name__=="__main__":
    model_path = "/runs/2/best.pt"
    FLOPS(model_path, "pets:")


    model = resnet50()
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input, ))
    print("----------------")
    print("resnet:")
    print("flops:",flops)
    print("params:",params)
    print("----------------")

    mobiel = mobilenet_v2()
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(mobiel, inputs=(input, ))
    print("----------------")
    print("mobile:")
    print("flops:",flops)
    print("params:",params)
    print("----------------")
