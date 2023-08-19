
# directory path of tensorflow DLLs in Windows
# somehow, the Conda environment can't read the required dlls when this path is included in the environment variables
INCLUDE_TF_DLL_PATH = True
TF_DLL_PATH = 'C:/Users/Subspace_Sig1/miniconda3/envs/denoiser/Library/bin'

# discriminater learning rate
D_LEARNING_RATE = 0.0001

# generater learning rate
G_LEARNING_RATE = 0.0001

# batch size
# if patching is disabled: make this a small value, e.g. 4, to avoid an out of memory (OOM) error
BATCH_SIZE = 4 #64

# batch shape
BATCH_SHAPE = [BATCH_SIZE, 224, 224, 3]

# enable patching
# setting this to False will most likely trigger an out of memory (OOM) error for large batch sizes
PATCH_ENABLE = False

# patch per image
PATCH_NUM = 50

# make sure it's a multiple of the image dimension to avoid annoying pre-processing padding during inference
PATCH_SIZE = BATCH_SHAPE[1] // 4

# path shape
PATCH_SHAPE = [BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 3]

# save discriminator
SAVE_DIS = True

# FIXME: rename to LOAD_WEIGHTS_ONLY
SAVE_WEIGHTS_ONLY = True

# loss weight factor
ADVERSARIAL_LOSS_FACTOR = 1.0
PIXEL_LOSS_FACTOR = 0.001
STYLE_LOSS_FACTOR = 0
SP_LOSS_FACTOR = 0.5
SMOOTH_LOSS_FACTOR = 0
SSIM_FACTOR = - 20.0
PSNR_FACTOR = - 2.0
D_LOSS_FACTOR = 1.0

# include csv report when running test denoising
GEN_CSV = True
