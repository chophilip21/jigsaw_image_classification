import torch.nn as nn
import torch
import pytorch_lightning as pl
from resnet import *
from utils import *
import torch.optim as optim
from pytorch_lightning.metrics.functional.classification import accuracy
from loss import *
# from jacobian import JacobianReg


# This is model is written considering L = 5 and S = 3 (so S + 1 steps)
class PMG(pl.LightningModule):

    def __init__(self, model, feature_size, lr, loss, classes_num, reg, batch_size=8, num_workers=6, root='bird'):
        super(PMG, self).__init__()

        self.features = model  
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)
        
        self.root = root
        self.reg = reg
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        """
        ----------------------------------------
        Refer to graph on page 6.
        This is Conv block L-2 and classfier L-2
        ----------------------------------------
        """
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
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
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
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
            BasicConv(self.num_ftrs, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
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

    # lightning will add optimizer inside the model
    def configure_optimizers(self):

        optimizer = optim.SGD([
            {'params': self.classifier_concat.parameters(), 'lr': self.lr},
            {'params': self.conv_block1.parameters(), 'lr': self.lr},
            {'params': self.classifier1.parameters(), 'lr': self.lr},
            {'params': self.conv_block2.parameters(), 'lr': self.lr},
            {'params': self.classifier2.parameters(), 'lr': self.lr},
            {'params': self.conv_block3.parameters(), 'lr': self.lr},
            {'params': self.classifier3.parameters(), 'lr': self.lr},
            {'params': self.features.parameters(), 'lr': self.lr/10}
        ],
            momentum=0.9, weight_decay=5e-4)
        
        # Learning rate optimizer options.
        cosineAnneal = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, verbose=True)
        #plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True) # you need to specify what you are monitoring. 

        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        warm_restart = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, verbose=True)
        
        return {'optimizer': optimizer, 'lr_scheduler': warm_restart}

    def training_step(self, batch, batch_idx):
      
        inputs, targets = batch
        loss_function = self.loss

        # step 1 (start from fine-grained jigsaw n=8)
        inputs1 = jigsaw_generator(inputs, 8)
        output_1, _, _, _ = self(inputs1) 
        loss1 = loss_function(output_1, targets) * 1 

        # step 2
        inputs2 = jigsaw_generator(inputs, 4)
        _, output_2, _, _ = self(inputs2)
        loss2 = loss_function(output_2, targets) * 1  

        # step 3
        inputs3 = jigsaw_generator(inputs, 2)
        _, _, output_3, _ = self(inputs3)
        loss3 = loss_function(output_3, targets) * 1

        # step 4 whole image, and vanilla loss. 
        _, _, _, output_concat = self(inputs)

        if self.reg == None:
            concat_loss = loss_function(output_concat, targets) * 2 
            train_loss = loss1 + loss2 + loss3 + concat_loss

        if self.reg == 'large_margin':
            pass

        elif self.reg == 'jacobian':
            pass

        # accuracy 
        _, predicted = torch.max(output_concat.data, 1)
        train_acc = accuracy(predicted, targets)

        metrics = {'loss': train_loss, 'accuracy': train_acc}

        self.log('accuracy', train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return metrics

    # * Not entirely the same as the training step. No jigsaw puzzle here.
    def validation_step(self, batch, batch_idx):

        inputs, targets = batch

        loss_function = self.loss
        
        #TODO: JOKER LOSS is probably not needed for validation
        output_1, output_2, output_3, output_concat = self(inputs)
        outputs_com = output_1 + output_2 + output_3 + output_3 + output_concat

        val_loss = loss_function(output_concat, targets)

        """
        There is the individual accuracy, and combined accuracy
        """
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)

        valid_acc = accuracy(predicted, targets)
        valid_acc_en = accuracy(predicted_com, targets)

        metrics = {'val_loss':  val_loss, 'val_acc': valid_acc, 'val_acc_en': valid_acc_en}

        self.log('val_acc', valid_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_en', valid_acc_en, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        return metrics

    
    def test_step(self, batch, batch_idx):

        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc' : metrics['val_acc_en'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics)


    def train_dataloader(self):

        transform_train = transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.RandomCrop(448, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.ImageFolder(
            root= self.root+'/train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        return trainloader

    def val_dataloader(self):

        transform_test = transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # ? Shuffle was True in the original implementation. This is most likely not the best practice.
        testset = torchvision.datasets.ImageFolder(root=self.root+'/test',
                                                   transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        return testloader


if __name__ == "__main__":

    print('model is working fine')
