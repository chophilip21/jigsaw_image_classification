# Pytorch Lightning implementation of Progressive Multi-Granularity (PMG) Learning for Fine-grained image classification

- This repository is Pytorch lightning implementation of https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training.
- This repository is created for SFU CMPT 757 class final project. 
- This repository has additional loss functions that can improve the results of the original code. 
- Refer to colab.pynb to see how things run exactly. 
- Follwing is the link for my report: https://drive.google.com/file/d/1TwjfJGza-sVAXdqWAxRbdO5NogSQxTRZ/view?usp=sharing

## Requirment

python 3.6

PyTorch >= 1.3.1

Pytorch lightning >= 1.10 

torchvision >= 0.4.2

## Instructions

- You can download dataset from the official link: http://www.vision.caltech.edu/visipedia/CUB-200.html
- Put your dataset (CUB-200-2011 is the default) and put it inside a dataset folder
- Run dataset.py and it will convert your dataset into following format:

```
dataset
├── train
│   ├── class_001
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   ├── class_002
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   └── ...
└── test
    ├── class_001
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    ├── class_002
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...

```

- If your dataset is not CUB_2011, refer to above structure or refer to my custom function. 

## Config 

- Model parameters are controlled by config.py. 
- The only thing that you should touch is  model type: ce_vanilla, ce_label_smooth, complement,
- The above parameter controls the loss function that you will be using. The vanilla function will just use cross entropy
- When you are training, make sure the number of classes match your dataset(i.e 200)

## Train
- Just run train.py
- Batch size will automatically adjust based on your computer capacity. Recommended size is 16.
- The performance will suffer if you cannot acheive batch size of 16.
- 16-bit precision is used. 

## weights

- The weights will auto-save to weights folder
- https://drive.google.com/file/d/1-0LvDBiODpgTMS9hw9HIwtFAwF-KYcuW/view?usp=sharing you can download the weights from here too. 
