import os
from PIL import Image
import numpy as np
from typing import List

class MaskCreator:
    SCENE_PATH = '../CloudNet/38-Cloud/Test/Entire_scene_gts'
    PREDICTION_PATH = 'predictions/'
    PATCH_SIZE = (192, 192)  # Model's patch size

    def __init__(self, gt_scene_path, scene_name):
        self.gt_scene_path = gt_scene_path
        self.scene_name = scene_name
        self.blank_scene = self.create_own_blank_scene()
        self.prediction_scene = None
        self.prediction_mask_path = os.path.join(os.path.join(MaskCreator.PREDICTION_PATH, self.scene_name), 'mask')


    def create_own_blank_scene(self):
        """
        Creates a blank mask with the same shape as the ground truth mask.
        """
        # Open the image to get its dimensions
        mask = Image.open(self.gt_scene_path)   

        width, height = mask.size

        # Calculate the new size (half the original size)
        new_width = width // 2
        new_height = height // 2

        # Resize the image
        resized_mask  = mask.resize(
            (new_width, new_height),
            resample=Image.Resampling.LANCZOS,  # High-quality downsampling
            reducing_gap=2.0  # Optimization by reducing the image in two steps
        )

        # save them in 'predictions/gt_scene/'
        os.makedirs(os.path.join(MaskCreator.PREDICTION_PATH, 'gt_scene'), exist_ok=True)
        resized_mask.save(os.path.join(MaskCreator.PREDICTION_PATH, 'gt_scene', self.scene_name + '.TIF'), compression="tiff_deflate", save_all=True, photometric='minisblack')

        mask = np.array(resized_mask)


        # Create a blank mask with the same shape
        blank_mask = np.zeros(mask.shape)

        return blank_mask
    

    def build_prediction_scene(self):
        """
        Paste each prediction mask onto the blank mask according to their row and column.
        """
        for filename in os.listdir(self.prediction_mask_path):
            # Extract row and column indices from the filename
            parts = filename.split('_')
            try:
                row_index = int(parts[2])
                col_index = int(parts[4])
            except ValueError:
                print(f"Filename {filename} does not match expected format.")
                continue

            # Calculate the position to paste the prediction mask
            top_left_y = ((row_index - 1) * MaskCreator.PATCH_SIZE[0] - 27)
            top_left_x = ((col_index - 1) * MaskCreator.PATCH_SIZE[1] - 63)

            file_path = os.path.join(self.prediction_mask_path, filename)

            # Load the prediction mask
            prediction_mask = np.array(Image.open(file_path))

            # Ensure the dimensions match before pasting
            if prediction_mask.shape != MaskCreator.PATCH_SIZE:
                print(f"Prediction mask shape {prediction_mask.shape} does not match expected size {MaskCreator.PATCH_SIZE}.")
                continue

            # Ensure the calculated position is within the bounds of the blank scene
            if top_left_y + MaskCreator.PATCH_SIZE[0] > self.blank_scene.shape[0] or top_left_x + MaskCreator.PATCH_SIZE[1] > self.blank_scene.shape[1]:
                print(f"Patch {filename} position out of bounds.")
                continue

            if top_left_y < 0 or top_left_x < 0:
                print(f"Patch {filename} position out of bounds.")
                continue

            # Paste the prediction mask onto the blank mask
            self.blank_scene[top_left_y:top_left_y + MaskCreator.PATCH_SIZE[0], top_left_x:top_left_x + MaskCreator.PATCH_SIZE[1]] = prediction_mask

        self.prediction_scene = self.blank_scene


    def convert_blank_to_image_and_save(self):
        prediction_path=os.path.join(MaskCreator.PREDICTION_PATH, 'predicted_scenes')
        self.prediction_scene = Image.fromarray(self.blank_scene.astype(np.uint8))
        # Ensure the directory exists
        os.makedirs(prediction_path, exist_ok=True)
        # Construct the file path
        prediction_scene_filename = self.scene_name + '.TIF'
        file_path = os.path.join(prediction_path, prediction_scene_filename)
        # Save the image
        self.prediction_scene.save(file_path)
        
        return file_path


    @staticmethod
    def create_prediction_scene() -> None:
        """
        Processes each scene in the specified directory, creates a mask creator for each,
        builds a prediction scene, and saves the result as an image.
        """
        mask_creators: List[MaskCreator] = [] 
        for scene in os.listdir(MaskCreator.SCENE_PATH):
            base_name: str = scene.split('.')[0]
            scene_name_parts: List[str] = base_name.split('_')
            scene_name: str = '_'.join(scene_name_parts[3:])
            gt_scene_path: str = os.path.join(MaskCreator.SCENE_PATH, scene)

            mask_creators.append(MaskCreator(gt_scene_path, scene_name))

        for mask_creator in mask_creators:
            print(f"Scene {mask_creator.scene_name} has shape {mask_creator.blank_scene.shape}")
            mask_creator.build_prediction_scene()
            mask_creator.convert_blank_to_image_and_save()