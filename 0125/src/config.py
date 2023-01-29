import torch

BATCH_SIZE = 24  # GPU Memory size
RESIZE_TO = 512  # resize the image training and transforms
NUM_EPOCHS = 100  # number of epochs to train for

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train image and xml files directory
TRAIN_DIR = '../Microcontroller Detection/train'
# validation image and xml files directory
VALID_DIR = '../Microcontroller Detection/test'

CLASSES = [
    'background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
]
NUM_CLASSES = 5

VISUALIZE_TRANSFORMED_ITEMS = False

OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2
