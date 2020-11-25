import os
from PIL import Image
import logging
import random
import torch
from resnet import *
from model import *
from utils import *


#! 512 is specific to feature size of ResNet50. 200 is for CUB_200. SHOULD be able to control this from config.py
def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 200)

    return net


if __name__== "__main__":  

    model_name = 'resnet50_pmg'
    model = load_model(model_name, pretrain=True, require_grad=True)
    print('The model has {:,} trainable parameters'.format(count_parameters(model)))
    
    """
    Call the dataloader here.
    - This is where we could implement better augmentation strategy (FastAutoAugment, etc)
    """

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    trainset = torchvision.datasets.ImageFolder(root='data/CUB_200_2011', transform=transform_train)

    print('==> Preparing Lightning Trainer..')



    # # debugging
    # trainer = pl.Trainer(fast_dev_run=True)
    # trainer.fit(model)

