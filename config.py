BATCH_SIZE = 2 # this will be automatically tuned and increased to best fit your machine.
NUM_WORKERS = 8
MODEL_BASE = 'resnet50_pmg'
LEARNING_RATE = 0.002 
CLASSES = 200
LOSS= 'agreement' #options: ce_vanilla, ce_label_smooth, ce_jacobian, agreement
EPOCH = 300