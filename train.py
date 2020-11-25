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



def load_model(model_name, pretrain=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        net = PMG(net, 512, 200, batch_size=BATCH_SIZE,
                  num_workers=NUM_WORKERS)

    return net


if __name__ == "__main__":

    # ! Tensorboard logging: tensorboard --logdir lightning_logs
    # ! refer to page 47 for info

    model_name = 'resnet50_pmg'

    if RESUME:
        model = torch.load(MODEL_PATH)
    else:
        print('training from scratch')
        model = load_model(model_name, pretrain=True)

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
                         max_epochs=EPOCH, gpus=1, benchmark=True, limit_train_batches=0.3) #input size is true so benchmark should give boost
    

    print('==> Starting the training process now...')

    # trainer.tune(model)
    trainer.fit(model)
