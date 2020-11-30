import numpy as np
import random
import torch
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.2):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss
"""
- These are all the losses used for calculating JoCOR loss
- It will only be used when co-reg is set to True 
- Simply setting LOSS = coreg will work
- Maximize agreement between classifier 
"""

def contrastive_loss(pred, second_pred, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(second_pred, dim=1), reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    
    else: 
        return torch.sum(kl, 1)


def jocor_loss(y_1, y_2, target, co_lambda=0.5):

    loss_pick_1 = F.cross_entropy(y_1, target, reduce=True) * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, target, reduce=True) * (1-co_lambda)

    joint_loss = (loss_pick_1 + loss_pick_2 + co_lambda * contrastive_loss(y_1, y_2, reduce=True) + co_lambda * contrastive_loss(y_2, y_1, reduce=True))

    return joint_loss
    
