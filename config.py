BATCH_SIZE = 2 #this will be automatically tuned and increased to best fit your machine.
NUM_WORKERS = 8
RESUME = False
MODEL_PATH = None
LEARNING_RATE = 0.0008
LOSS= 'ce_vanilla' #other options should be ce_label_smooth, co-regularization
EPOCH = 300