# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -5.2
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 5
lamda_guide = 5
lamda_low = 2
lamda_high = 1

device_ids = [0]

# Train:
batch_size = 8

cropsize = 224


cropsize_val = 1024

betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 1024
batchsize_val = 2
shuffle_val = False
val_freq = 1


# Dataset
TRAIN_PATH = r'C:\Users\cze\Desktop\data\train/'
VAL_PATH = r'C:\Users\cze\Desktop\data\test/'
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = r'C:\Users\cze\Desktop\我的论文\第三篇\3\MSIHNet\model/'
checkpoint_on_error = True
SAVE_freq = 1

IMAGE_PATH = '/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'

# Load:
suffix = 'model_checkpoint_00010.pt'
tain_next = False
trained_epoch = 0
