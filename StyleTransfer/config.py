
project_path = "C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer"

model_path = "\\data\\model"
checkpoints_path = "\\data\\checkpoints"
images_path = "\\data\\images"

log_train = "\\logs\\training_network"
log_generator = "\\logs\\generator_network"


output_generator = "\\outputs\\generator_networks"
output_images = "\\outputs\\images"

VGG_STYLE_TENSOR_1 = "import/conv1_2/Relu:0"
VGG_STYLE_TENSOR_2 = "import/conv2_2/Relu:0"
VGG_STYLE_TENSOR_3 = "import/conv3_2/Relu:0"
VGG_STYLE_TENSOR_4 = "import/conv4_2/Relu:0"
VGG_INPUT_RESOLUTION = 224

VGG_CONTENT_LAYER = "import/conv2_2/Relu:0"

INIT_STD_DEV = .1
TRUNCATED_SEED = 1



BATCH_SIZE = 4
PRECOMPUTE_BATCH_SIZE = 20

SEED = 448

DOWN_SAMPLING = 2
INPUT_RESOLUTION = 224
LEARNING_RATE = 0.001

CONTENT_WEIGHT = 7.5
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2


CHICAGO_WIDTH = 712
CHICAGO_RATIO = 119.0/178.0

ANDROID_WIDTH = 468
ANDROID_RATIO = 16.0/9.0

ANDROID_EMULATOR_WIDTH = 304
ANDROID_EMULATOR_RATIO = 1.0

FULL_HD_WIDTH = 1920
FULL_HD_RATIO = 9.0/16.0