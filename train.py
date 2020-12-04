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
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from loss import *

def load_model(model_name, loss, learning_rate, batch_size, num_workers, regularizer, pretrain=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        net = PMG(net, loss=loss, feature_size = 512, classes_num = CLASSES, batch_size=batch_size,
                  num_workers=num_workers, lr = learning_rate, reg=regularizer, root='bird') ## this should work right?

    return net


if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)
    # torch.set_deterministic(True)

    model_name = MODEL_BASE
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = 'weights/{}/{}'.format(time,LOSS)
    reg_type = None

    if LOSS == 'ce_vanilla':
        print('==> The loss function is set as: ', LOSS)
        loss = nn.CrossEntropyLoss()

    elif LOSS == 'ce_label_smooth':
        print('==> The loss function is set as: ', LOSS)
        loss = SmoothCrossEntropyLoss()

    elif LOSS == 'large_margin':
        loss = nn.CrossEntropyLoss()
        reg_type = 'large_margin'

    elif LOSS == 'complement':
        loss = ComplementCrossEntropy()

    else:
        print('====> The LOSS IS NOT SET PROPERLY')
            
    model = load_model(model_name, loss, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS, pretrain=True, regularizer=reg_type)

    print('The model has {:,} trainable parameters'.format(
        count_parameters(model)))


    """
    ------------------------------------
    Intialize the trainier from the model and the callbacks
    # Tensorboard logging: tensorboard --logdir lightning_logs
    ------------------------------------
    """
    
    bar = ProgressBar()
    # early_stopping = EarlyStopping('val_acc_en', patience=15)
    
    # Two validation metrics. Let's have two different saver. 
    ckpt_en = ModelCheckpoint(
        dirpath=save_dir, monitor='val_acc_en', mode = 'auto', save_last=True, filename='{epoch:02d}-{val_acc_en:.4f}')

    ckpt_reg = ModelCheckpoint(
        dirpath=save_dir, monitor='val_acc', mode = 'auto', save_last=True, filename='{epoch:02d}-{val_acc:.4f}')
    

    csv_logger = CSVLogger('csv_log/{}'.format(time), name = '{}_model'.format(LOSS))
    tensorboard_logger = TensorBoardLogger('tb_log/{}'.format(time), name='{}_model'.format(LOSS))

    # use resume_from_checkpoint
    # resume_point = 'weights/20201201_180420/ce_label_smooth/last.ckpt'

    trainer = pl.Trainer(auto_scale_batch_size='power', callbacks=[bar, ckpt_en, ckpt_reg],
                         max_epochs=EPOCH, gpus=1, precision=16, 
                         accumulate_grad_batches=1, logger = [tensorboard_logger, csv_logger])
    
    print('==> Starting the training process now...')

    trainer.tune(model) 
    trainer.fit(model)

    # trainer.test()
