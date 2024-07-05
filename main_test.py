import os
import tensorflow as tf
import tifffile as tiff
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

import utils
from dataset import CSVLoader
from dataset import ImageDataset, DeepLabV3DataGenerator
from augmentation import DataTransformer
from accuracy import run_accuracy
from mask_creator import MaskCreator

GLOBAL_PATH = '../CloudNet/Full_Cloud/'
TEST_PATH = os.path.join(GLOBAL_PATH, 'test/')
TRAIN_CSV = os.path.join(TEST_PATH, 'test_set.csv')
SAVE_PATH = 'saved_models/'
PREDICTION_PATH = 'predictions/'

def create_test_val_generators():

    test_csv_loader = CSVLoader(TRAIN_CSV)
    test_image_paths, _ = test_csv_loader.load_images_path(TEST_PATH, is_training=False)

    # Load ground truth mask paths

    # Create datasets
    test_dataset = ImageDataset(test_image_paths)
    # ground_truth_dataset = ImageDataset(ground_truth_paths)

    # Create transformer (without augmentation for test data)
    transformer = DataTransformer(img_rows=192, img_cols=192, max_possible_input_value=65535)

    # Create generators
    test_generator = DeepLabV3DataGenerator(test_dataset, transformer, batch_size=16, is_training=False)
    # ground_truth_generator = DeepLabV3DataGenerator(ground_truth_dataset, transformer, batch_size=32, is_training=False)
    return test_generator


def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

def simple_test():
    path = '../CloudNet/38-Cloud/38-Cloud_training/'

    image_blue = os.path.join(path, 'train_blue/blue_patch_6_1_by_6_LC08_L1TP_032029_20160420_20170223_01_T1.TIF')
    image_green = os.path.join(path, 'train_green/green_patch_6_1_by_6_LC08_L1TP_032029_20160420_20170223_01_T1.TIF')
    image_red = os.path.join(path, 'train_red/red_patch_6_1_by_6_LC08_L1TP_032029_20160420_20170223_01_T1.TIF')
    image_nir = os.path.join(path, 'train_nir/nir_patch_6_1_by_6_LC08_L1TP_032029_20160420_20170223_01_T1.TIF')

    image = ImageDataset.load_image((image_blue, image_green, image_red, image_nir))

    gt = ImageDataset.load_mask(os.path.join(path, 'train_gt/gt_patch_6_1_by_6_LC08_L1TP_032029_20160420_20170223_01_T1.TIF'))

    transformer = DataTransformer(img_rows=192, img_cols=192, max_possible_input_value=65535)

    image = transformer.transform_image(image)
    gt = transformer.transform_mask(gt)
    
    return image, gt


def one_image_mask():
    model = load_model('saved_models/best_unet.h5')

    image, gt_mask = simple_test()
    image = np.expand_dims(image, axis=0)
    print(image.shape)

    prediction = model.predict(image, verbose=1)


    prediction = np.squeeze(prediction, axis=0)
    image = np.squeeze(image, axis=0)
    

    print(prediction.shape)
    print(gt_mask.shape)
    # Save the image and prediction
    pred_mask_binary = (prediction > 0.5).astype(np.uint8)

    
    utils.save_image_mask(image, prediction, 'image.tif', 'mask.tif')

def predict_whole_set():

    test_generator = create_test_val_generators()
    model = load_model('saved_models/best_unet.h5')
    # Get first image from the test generator
    for batch_images, batch_filenames in tqdm(test_generator, desc="Processing batches"):
        # Predict masks for the batch of images
        batch_predictions = model.predict(batch_images)

        # Iterate through each image, prediction, and filename in the batch
        for image, prediction, filename in zip(batch_images, batch_predictions, batch_filenames):
            # Extract the scene identifier from the filename
            scene_identifier = utils.extract_scene_ids(filename)  # Adjust this based on your filename format

            # Create a directory for the scene if it doesn't exist
            scene_dir = os.path.join('predictions', scene_identifier)
            os.makedirs(scene_dir, exist_ok=True)
            os.makedirs(os.path.join(scene_dir, 'image'), exist_ok=True)
            os.makedirs(os.path.join(scene_dir, 'mask'), exist_ok=True)

            # Define paths for saving the image and its mask
            image_path = os.path.join(os.path.join(scene_dir, 'image'), f"{filename}.TIF")
            mask_path = os.path.join(os.path.join(scene_dir, 'mask'), f"{filename}.TIF")

            # Save the original image and its predicted mask
            pred_mask_binary = (prediction > 0.5).astype(np.uint8)
            utils.save_image_mask(image, pred_mask_binary, image_path, mask_path)






def main():
    predict_whole_set()
    # MaskCreator.create_prediction_scene()
    # prediction_scene_path = os.path.join(MaskCreator.PREDICTION_PATH, 'predicted_scenes')
    # gt_scene_path = os.path.join(MaskCreator.PREDICTION_PATH, 'gt_scene')
    # run_accuracy(gt_scene_path, prediction_scene_path)
   




if __name__ == "__main__":
    main()