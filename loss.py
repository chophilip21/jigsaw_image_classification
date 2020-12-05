import numpy as np
import random
import torch
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss

class SmoothCrossEntropyLoss(_WeightedLoss):

    """
    This applies simple label smoothing to the cross entropy with strength parameter of 0.2
    """

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

class LargeMarginInSoftmaxLoss(nn.CrossEntropyLoss):
    """
    This combines the Softmax Cross-Entropy Loss (nn.CrossEntropyLoss) and the large-margin inducing
    """
    def __init__(self, reg_lambda=0.1, deg_logit=None, 
                weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(LargeMarginInSoftmaxLoss, self).__init__(weight=weight, size_average=size_average, 
                                ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.reg_lambda = reg_lambda
        self.deg_logit = deg_logit

    def forward(self, input, target):
        N = input.size(0) # number of samples
        C = input.size(1) # number of classes
        Mask = torch.zeros_like(input, requires_grad=False)
        Mask[range(N),target] = 1

        if self.deg_logit is not None:
            input = input - self.deg_logit * Mask
        
        loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        X = input - 1.e6 * Mask # [N x C], excluding the target class
        reg = 0.5 * ((F.softmax(X, dim=1) - 1.0/(C-1)) * F.log_softmax(X, dim=1) * (1.0-Mask)).sum(dim=1)
        if self.reduction == 'sum':
            reg = reg.sum()
        elif self.reduction == 'mean':
            reg = reg.mean()
        elif self.reduction == 'none':
            reg = reg

        return loss + self.reg_lambda * reg


class ComplementEntropy(nn.Module):
    '''Compute the complement entropy of complement classes.'''
    def __init__(self, num_classes=200):
        super(ComplementEntropy, self).__init__()
        self.classes = num_classes
        self.batch_size = None

    def forward(self, y_hat, y):
        self.batch_size = len(y)
        y_hat = F.softmax(y_hat, dim=1)
        Yg = torch.gather(y_hat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7
        Px = y_hat / Yg_.view(len(y_hat), 1)
        Px_log = torch.log(Px + 1e-10)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_\
            (1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        entropy = torch.sum(output)
        entropy /= float(self.batch_size)
        entropy /= float(self.classes)
        return entropy


class ComplementCrossEntropy(nn.Module):
    def __init__(self, num_classes=200, gamma=5):
        super(ComplementCrossEntropy, self).__init__()
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss()
        self.complement_entropy = ComplementEntropy(num_classes)

    def forward(self, y_hat, y):
        l1 = self.cross_entropy(y_hat, y)
        l2 = self.complement_entropy(y_hat, y)
        return l1 + self.gamma * l2



class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def nansum(self, x):
        return x[~torch.isnan(x)].sum()

    def forward(self, y_hat, y):

        conditional = y_hat/y
        cond_ent = - y_hat * torch.log(conditional)

        out = self.nansum(cond_ent)

        return out


class MaximumEntropy(nn.Module):

    def __init__(self, gamma=1):
        super(MaximumEntropy, self).__init__()
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss()
        self.entropy = Entropy()

    def forward(self, y_hat, y):
        l1 = self.cross_entropy(y_hat, y)
        l2 = self.entropy(y_hat, y)
        return l1 - (self.gamma * l2)




    
