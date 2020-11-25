import torch.nn as nn
import torch
import pytorch_lightning as pl
from resnet import *
from utils import *


"""
This is model is written considering L = 5 and S = 3 (so S + 1 steps)
"""
class PMG(pl.LightningModule):
    
    def __init__(self, model, feature_size, classes_num):
        super(PMG, self).__init__()

        self.features = model #! This is ResNet50 where L=5  
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True) 
     
        """
        ----------------------------------------
        Refer to graph on page 6.
        This is Conv block L-2 and classfier L-2
        ----------------------------------------
        """
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        """
        -----------------------------------------
        This is Conv block L-1 and classfier L-1
        -----------------------------------------
        """
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )  

        self.classifier2 = nn.Sequential(
        nn.BatchNorm1d(self.num_ftrs//2),
        nn.Linear(self.num_ftrs//2, feature_size),
        nn.BatchNorm1d(feature_size),
        nn.ELU(inplace=True),
        nn.Linear(feature_size, classes_num),
        )

        """
        ----------------------------------------
        This is Conv block L and classfier L
        ----------------------------------------
        """
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        """
        ----------------------------------------
        Refer to graph on page 6 and 7.
        This is the classifier concat
        ----------------------------------------
        """
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x):

        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)

        """
        xcl, xc2, xc3 will be used to calculate the loss.
        x_concat is just concat of layer features (1,2,3) 
        """
        x_concat = torch.cat((xl1, xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)
        return xc1, xc2, xc3, x_concat



    #! Notice the author feeds the input into nn.DataParallel(model, device_ids=[0,1]), which is probably not needed for Lightning
    def training_step(self, batch, batch_idx):
        
        """
        This method is reserved for lightning
        - No need for backward(), step(), etc. 
        - Here, Image I(n=1) becomes I(n=8), I(n=4), I(n=2), I(n=1). Loss calculated form each of them.
        """

        inputs, targets = batch
        CELoss = nn.CrossEntropyLoss()

        # step 1 (start from fine-grained jigsaw n=8)
        inputs1 = jigsaw_generator(inputs, 8) 
        output_1, _, _, = self(inputs1) # todo: check this is right
        loss1 = CELoss(output_1, targets) * 1 # alpha =1

        # step 2 
        inputs2 = jigsaw_generator(inputs, 4)
        _, output_2, _, _ = self(inputs2)
        loss2 = CELoss(output_2, targets) * 1 #alpha = 1

        # step 3
        inputs3 = jigsaw_generator(inputs, 2)
        _, _, output_3, _ = self(inputs3)
        loss3 = CELoss(output_3, targets) * 1

        """ step 4 (final step). You do not use jigsaw here, as you are using the image itself """
        _, _, _, output_concat = self(inputs)
        concat_loss = CELoss(output_concat, targets) * 2 # beta = 2

        # Todo: skipping all details regarding accuracy. Make sure below detail is correct
        train_loss = loss1.item() + loss2.item() + loss3.item() + concat_loss.item()
        # train_loss = train_loss / (batch_idx + 1) #! I am NOT going to divide by batch index

        return {'loss' :  train_loss}

       









    
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





if __name__== "__main__":  

    print('model is working fine')