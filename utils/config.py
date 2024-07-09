import os
import cv2

class Config:
    # Paths
    GLOBAL_PATH = '../CloudNet/Full_Cloud/'
    TRAIN_PATH = os.path.join(GLOBAL_PATH, 'train/')
    TEST_PATH = os.path.join(GLOBAL_PATH, 'test/')
    TRAIN_CSV = os.path.join(TRAIN_PATH, 'train_set.csv')
    TEST_CSV = os.path.join(TEST_PATH, 'test_set.csv')
    TRAIN_GT_DIR = 'train_gt'
    TEST_GT_DIR = 'test_gt_resized'

    # Output paths
    OUTPUT_DIR = 'output/'
    SAVE_PATH = os.path.join(OUTPUT_DIR, 'saved_models/')
    PREDICTION_PATH = os.path.join(OUTPUT_DIR, 'predictions/')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs/')
    METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics/')
    OVERALL_METRICS_PATH = os.path.join(METRICS_DIR, 'overall_metrics.csv')

    # Model properties
    UNET_PATH = os.path.join(SAVE_PATH, 'best_unet.keras')
    MODEL_INPUT_SHAPE = (192, 192, 4)
    BATCH_SIZE = 12
    NUM_CLASSES = 1



    # Image properties
    MAX_PIXEL_VALUE = 65535
    IMAGE_SIZE = (192, 192)
    CHANNELS = ['red', 'green', 'blue', 'nir']
    TRAIN_DIR_PATH = 'train'
    DIR_TYPE_NAME = {True: "train", False: "test"}
    GT_PREFIX = 'gt_'


    # Magic strings
    IMAGE = 'image'
    MASK = 'mask'
    IMAGE_EXTENSION = '.TIF'
    MASK_EXTENSION = '.TIF'

    
    # Model training
    MAX_EPOCHS = 2000
    INITIAL_LEARNING_RATE = 0.001

    # Image properties
    NUM_CHANNELS = len(CHANNELS)

    