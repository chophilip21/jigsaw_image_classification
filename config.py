BATCH_SIZE = 2 #this will be automatically tuned and increased to best fit your machine.
NUM_WORKERS = 8
MODEL_BASE = 'resnet50_pmg'
MODEL_PATH = None
LEARNING_RATE = 0.002 # 0.0008 was not bad
CLASSES = 200
LOSS= 'ce_vanilla' #other options should be ce_vanilla, ce_label_smooth, coreg
EPOCH = 300