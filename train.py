import os
from PIL import Image
import logging
import random
from datetime import datetime
from resnet import *
from model import *
from utils import *
from config import *
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ProgressBar, GradientAccumulationScheduler, ModelCheckpoint, EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from loss import *
# from pytorch_lightning import seed_everything
# ! Tensorboard logging: tensorboard --logdir lightning_logs

def load_model(model_name, loss, learning_rate, batch_size, num_workers, pretrain=True, coreg=None):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        net = PMG(net, loss=loss, feature_size = 512, classes_num = 200, batch_size=batch_size,
                  num_workers=num_workers, lr = learning_rate, coreg=coreg)

    return net


if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)

    model_name = 'resnet50_pmg'
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = 'weights/{}/{}'.format(time,LOSS)

    if LOSS == 'ce_vanilla':
        print('==> The loss function is set as: ', LOSS)
        loss = nn.CrossEntropyLoss()

    elif LOSS == 'ce_label_smooth':
        print('==> The loss function is set as: ', LOSS)
        loss = nn.CrossEntropyLoss()

    # If Coreg == True, we should calculate something different. 
    elif LOSS == 'coreg':
        pass


    model = load_model(model_name, loss, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS, pretrain=True, coreg=None)

    print('The model has {:,} trainable parameters'.format(
        count_parameters(model)))


    """
    ------------------------------------
    Intialize the trainier from the model and the callbacks
    ------------------------------------
    """
    
    bar = ProgressBar()
    early_stopping = EarlyStopping('val_acc_en', patience=15)
    ckpt = ModelCheckpoint(
        dirpath=save_dir, monitor='val_acc_en', mode = 'auto', save_last=True, filename='{epoch:02d}-{val_acc_en:.4f}')
    tensorboard_logger = TensorBoardLogger('tb_log/{}'.format(time), name='{}_model'.format(LOSS))

    # You can resume training with resume_from_checkpoint
    # resume_point = 'weights/20201128_173524/ce_vanilla/epoch=50-val_acc_en=0.8642.ckpt'

    trainer = pl.Trainer(auto_scale_batch_size='power', callbacks=[bar, ckpt],
                         max_epochs=EPOCH, gpus=1, precision=16, 
                         accumulate_grad_batches=2, logger = tensorboard_logger)
    
    print('==> Starting the training process now...')

    trainer.tune(model) 
    trainer.fit(model)

    # trainer.test()