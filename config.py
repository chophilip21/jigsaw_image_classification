BATCH_SIZE = 2 # this will be automatically tuned and increased to best fit your machine.
NUM_WORKERS = 6
MODEL_BASE = 'resnet50_pmg'
LEARNING_RATE = 0.0008
CLASSES = 200
LOSS= 'complement' #options: ce_vanilla, ce_label_smooth, complement, large_margin
EPOCH = 300