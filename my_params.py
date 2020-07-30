from my_utils.image_preprocess import HZ_preprocess

### COMMON ###
INPUT_SIZE = 250
INPUT_CHANNELS = 3

### INFERENCE MODEL ###
HZ_preprocess_activate = False
image_preprocess_func = HZ_preprocess
SEC_BETWEEN_PREDICTION = 1.5

### EXPORT MODEL ###
EXPORT_MODEL_PATH = './deployed/MobileNet_V3_large_ISIC_ver1.pb'
CHECKPOINT_DIR = './train_log/30_07_2020_00_40_45'

### EXPORT TFLITE ###
graph_def_file='./deployed/MobileNet_V3_large_ver2.pb'
input_arrays='ImageTensor'
output_arrays='SemanticPredictions'
input_shape='1,{0},{0},{1}'.format(INPUT_SIZE, INPUT_CHANNELS)
inference_input_type='QUANTIZED_UINT8'
inference_type='FLOAT'
mean_values=128
std_dev_values=127