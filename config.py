BATCH_SIZE = 2 # this will be automatically tuned and increased to best fit your machine.
NUM_WORKERS = 6
MODEL_BASE = 'resnet50_pmg'
LEARNING_RATE = 0.0008
CLASSES = 200
LOSS= 'ce_vanilla' #options: ce_vanilla, ce_label_smooth, complement,
EPOCH = 300