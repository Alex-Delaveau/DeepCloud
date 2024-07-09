import numpy as np
import cv2
from skimage import transform
from typing import Tuple
from utils.config import Config

class DataTransformer:
    def __init__(self, img_shape = Config.IMAGE_SIZE, max_possible_input_value: float = Config.MAX_PIXEL_VALUE):
        self.img_rows = img_shape[0]
        self.img_cols = img_shape[1]
        self.max_possible_input_value = max_possible_input_value

    def transform_image(self, image):
        
        # Ensure image is a numpy array
        if not isinstance(image, np.ndarray):
            try:
                image = np.array(image)
            except:
                raise ValueError(f"Unable to convert image to numpy array. Image type: {type(image)}")
        
        # Check if image is grayscale (2D) and convert to 3D if necessary
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) != 3:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Ensure image is in the correct data type
        image = image.astype(np.float32)
        
        # Resize the image
        try:
            image = cv2.resize(image, (self.img_cols, self.img_rows), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Error during resize: {str(e)}")
            print(f"Image shape before resize: {image.shape}")
            raise
        
        # Normalize the image
        return image / self.max_possible_input_value

    def transform_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.resize(mask, (self.img_cols, self.img_rows), interpolation=cv2.INTER_NEAREST)
        return mask[..., np.newaxis].astype(np.float32) / 255

    def rotate_clk(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        angle = np.random.choice([4, 6, 8, 10, 12, 14, 16, 18, 20])
        img_o = transform.rotate(img, angle, resize=False, preserve_range=True, mode='symmetric')
        msk_o = transform.rotate(msk, angle, resize=False, preserve_range=True, mode='symmetric')
        return img_o, msk_o

    def rotate_cclk(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        angle = np.random.choice([-20, -18, -16, -14, -12, -10, -8, -6, -4])
        img_o = transform.rotate(img, angle, resize=False, preserve_range=True, mode='symmetric')
        msk_o = transform.rotate(msk, angle, resize=False, preserve_range=True, mode='symmetric')
        return img_o, msk_o

    def flip(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_o = np.flip(img, axis=1)
        msk_o = np.flip(msk, axis=1)
        return img_o, msk_o

    def zoom(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        zoom_factor = np.random.choice([1.2, 1.5, 1.8, 2, 2.2, 2.5])
        h, w = img.shape[:2]

        zh = int(np.round(zoom_factor * h))
        zw = int(np.round(zoom_factor * w))

        img_zoomed = transform.resize(img, (zh, zw), preserve_range=True, mode='symmetric')
        msk_zoomed = transform.resize(msk, (zh, zw), preserve_range=True, mode='symmetric')
        region = np.random.choice([0, 1, 2, 3, 4])

        if zoom_factor <= 1:
            outimg, outmsk = img_zoomed, msk_zoomed
        else:
            if region == 0:
                outimg, outmsk = img_zoomed[0:h, 0:w], msk_zoomed[0:h, 0:w]
            elif region == 1:
                outimg, outmsk = img_zoomed[0:h, zw - w:zw], msk_zoomed[0:h, zw - w:zw]
            elif region == 2:
                outimg, outmsk = img_zoomed[zh - h:zh, 0:w], msk_zoomed[zh - h:zh, 0:w]
            elif region == 3:
                outimg, outmsk = img_zoomed[zh - h:zh, zw - w:zw], msk_zoomed[zh - h:zh, zw - w:zw]
            elif region == 4:
                marh, marw = h // 2, w // 2
                outimg = img_zoomed[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw)]
                outmsk = msk_zoomed[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw)]

        img_o = transform.resize(outimg, (h, w), preserve_range=True, mode='symmetric')
        msk_o = transform.resize(outmsk, (h, w), preserve_range=True, mode='symmetric')
        return img_o, msk_o

    def apply_random_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        augmentations = [self.rotate_clk, self.rotate_cclk, self.flip, self.zoom]
        aug_func = np.random.choice(augmentations)
        
        return aug_func(image, mask)

    def transform_and_augment(self, image: np.ndarray, mask: np.ndarray, apply_augmentation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if apply_augmentation:
            image, mask = self.apply_random_augmentation(image, mask)
        
        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        
        return image, mask