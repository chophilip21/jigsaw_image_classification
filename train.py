import os
from PIL import Image
import logging
import random
import torch
from pytorch_lightning.callbacks import ProgressBar, GradientAccumulationScheduler, ModelCheckpoint, EarlyStopping
from resnet import *
from model import *
from utils import *
from config import *
from pytorch_lightning.tuner.tuning import Tuner



#! 512 is specific to feature size of ResNet50. 200 is for CUB_200. SHOULD be able to control this from config.py
def load_model(model_name, pretrain=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        net = PMG(net, 512, 200, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    return net


if __name__ == "__main__":

    """
    Tensorboard logging: tensorboard --logdir lightning_logs
    """

    model_name = 'resnet50_pmg'

    if RESUME:
        model = torch.load(MODEL_PATH)
    else:
        model = load_model(model_name, pretrain=True)

    print('The model has {:,} trainable parameters'.format(
        count_parameters(model)))


    """
    ------------------------------------
    Intialize the trainier from the model and the callbacks
    ------------------------------------
    """

    save_dir = 'weights'
    bar = ProgressBar()
    accumulator = GradientAccumulationScheduler(scheduling={5: 2})
    early_stopping = EarlyStopping('val_loss', patience=15)
    ckpt = ModelCheckpoint(dirpath=save_dir, monitor='val_loss', filename='{epoch:03d}')

    print('==> Preparing Lightning Trainer..')
    trainer = pl.Trainer(auto_scale_batch_size='binsearch', callbacks = [bar,accumulator, ckpt], max_epochs=EPOCH)
    # trainer.tune(model) #! Fine tuning gave batch_size of 8 as the maximum for me. 
    # trainer.fit(model)


