DEVICE = 'cuda'

DATA_DIR_1 = "C:\\Users\\user\\Data\\700_photos"
DATA_DIR_2 = "C:\\Users\\user\\Data\\300_photos"
DATA_DIR_3 = "C:\\Users\\user\\Data\\real_photos"

NUM_CLASSES = 3
SCALE = 100

config_train = {
    'dataset_format': 'coco',
    'path': DATA_DIR_1,
    'splits': {
        'train': ('result.json', 'train'),
    },
}


config_val = {
    'dataset_format': 'coco',
    'path': DATA_DIR_2,
    'splits': {
        'train': ('result.json', 'valid'),
    },
}

config_test = {
    'dataset_format': 'coco',
    'path': DATA_DIR_3,
    'splits': {
        'train': ('result.json', 'test'),
    },
}

