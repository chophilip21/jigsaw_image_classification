from __future__ import print_function
import os
import sys
import time
import logging
import numpy as np


#! Probably this is just parameter for training
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    print('things are working fine')
    # print('The model has {:,} trainable parameters'.format(count_parameters(model)))
