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
from datetime import datetime



def load_model(model_name, learning_rate, batch_size, num_workers, pretrain=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        net = PMG(net, feature_size = 512, classes_num = 200, batch_size=batch_size,
                  num_workers=num_workers, lr = learning_rate)

    return net


if __name__ == "__main__":

    # ! Tensorboard logging: tensorboard --logdir lightning_logs
    # ! refer to page 47 for info

    model_name = 'resnet50_pmg'

    if RESUME:
        model = torch.load(MODEL_PATH)
    else:
        print('training from scratch')
        model = load_model(model_name, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS, pretrain=True)

    print('The model has {:,} trainable parameters'.format(
        count_parameters(model)))

    """
    ------------------------------------
    Intialize the trainier from the model and the callbacks
    ------------------------------------
    """
    
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = 'weights/{}'.format(time)

    # callbacks
    bar = ProgressBar()
    accumulator = GradientAccumulationScheduler(scheduling={5: 3, 10: 5})
    early_stopping = EarlyStopping('val_loss', patience=15)
    ckpt = ModelCheckpoint(
        dirpath=save_dir, monitor='val_loss', mode = 'auto', save_last=True, filename='{epoch:02d}-{val_acc_en:.2f}')

    trainer = pl.Trainer(auto_scale_batch_size='power', callbacks=[bar, accumulator, ckpt],
                         max_epochs=2, gpus=1, precision=16)
    

    print('==> Starting the training process now...')

    trainer.tune(model) #! Fine tuning
    trainer.fit(model)
