import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import utils.utils as utils
from utils.config import Config
from data.dataset import ImageDataset, DeepLabV3DataGenerator, CSVLoader
from data.augmentation import DataTransformer
from accuracy.accuracy import create_coupled_path, run_accuracy_on_couples_path


def create_test_val_generators():

    test_csv_loader = CSVLoader(Config.TEST_CSV)
    test_image_paths, _ = test_csv_loader.load_images_path(Config.TEST_PATH, is_training=False)


    # Create datasets
    test_dataset = ImageDataset(test_image_paths)

    # Create transformer (without augmentation for test data)
    transformer = DataTransformer()

    # Create generators
    test_generator = DeepLabV3DataGenerator(test_dataset, transformer, is_training=False)
    return test_generator


def predict_whole_set(model, generator, path=Config.PREDICTION_PATH):

    # Get first image from the test generator
    for batch_images, batch_filenames in tqdm(generator, desc="Processing batches"):
        # Predict masks for the batch of images
        print(batch_images.shape)
        batch_predictions = model.predict(batch_images)
        print(batch_predictions.shape)

        # Iterate through each image, prediction, and filename in the batch
        for image, prediction, filename in zip(batch_images, batch_predictions, batch_filenames):
            # Extract the scene identifier from the filename
            scene_identifier = utils.extract_scene_ids(filename)  # Adjust this based on your filename format

            # Create a directory for the scene if it doesn't exist
            scene_dir = os.path.join(path, scene_identifier)
            print(scene_dir)
            os.makedirs(scene_dir, exist_ok=True)
            os.makedirs(os.path.join(scene_dir, Config.IMAGE), exist_ok=True)
            os.makedirs(os.path.join(scene_dir, Config.MASK), exist_ok=True)

            # Define paths for saving the image and its mask
            image_path = os.path.join(os.path.join(scene_dir, Config.IMAGE), f"{filename}{Config.IMAGE_EXTENSION}")
            mask_path = os.path.join(os.path.join(scene_dir, Config.MASK), f"{filename}{Config.MASK_EXTENSION}")

            # Save the original image and its predicted mask
            utils.save_image_mask(image, prediction, image_path, mask_path)


def main():
    models = {
        'unet' : utils.load_model(Config.UNET_PATH), 
        # 'cloudnet' : utils.load_model(Config.CLOUDNET_PATH), 
        # 'deeplab' : utils.load_model(Config.DEEPLAB_PATH)
        }
    # UNET
    for model_name, model in models.items():
        print(f"Running predictions for {model_name}")
        generators = create_test_val_generators()
        model_path = os.path.join(Config.PREDICTION_PATH, model_name)
        predict_whole_set(model, generators, model_path)
        # coupled_paths = create_coupled_path(model_path)
        # run_accuracy_on_couples_path(coupled_paths, os.path.join(Config.METRICS_DIR, model_name))


if __name__ == "__main__":
    main()