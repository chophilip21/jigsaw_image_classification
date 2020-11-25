''' CONSIDER THIS LATER ON

ROOT = 'data/CUB_200_2011'
MODEL_NAME = 'resnet50_pmg'


if MODEL_NAME == 'resnet50_pmg':
    NUM_FTS = 512

if ROOT == 'data/CUB_200_2011':
    NUM_CLS = 200
elif ROOT == 'data/stanford_car':
    NUM_CLS = 196
else:
    print('I do not know what you want me to do')