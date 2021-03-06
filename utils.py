import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from PIL import Image


#! Not using this
def cosine_anneal_schedule(t, nb_epoch, lr):
    # t - 1 is used when t has 1-based indexing.
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()

    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                           y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size,
                y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def save_image(data, filename):
        img = data.clone().clamp(0, 255).numpy()
        img = img[0].transpose(1, 2, 0)
        img = Image.fromarray(img, mode='RGB')
        img.save(filename)


# This probably doesn't have to be a lightning module
class BasicConv(pl.LightningModule):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




if __name__ == "__main__":

    print('Testing jigsaw on a random image')

    img = Image.open('example.jpg')

    size = 448, 448
    img.thumbnail(size)
    img.save('resized.jpg')

    image = Image.open('resized.jpg')

    tensor = transforms.ToTensor()(image)
    print(tensor.shape)

    test = jigsaw_generator(tensor, 2)

    test = transforms.ToPILImage()(test)

    test.show()
