import numpy as np
import cv2 
import tensorflow as tf
from typing import List, Tuple, Optional
from augmentation import DataTransformer
import pandas as pd
from tqdm.auto import tqdm
from utils import extract_filename

class ImageDataset:
    def __init__(self, image_paths: List[Tuple[str, str, str, str]], mask_paths: List[str] = None):
        """
        Initialize the dataset with image and optional mask paths.
        
        :param image_paths: List of tuples, each containing paths to R, G, B, NIR channel images
        :param mask_paths: List of paths to mask images (optional, for training data)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_training = mask_paths is not None
        self.filenames = [extract_filename(paths[0]) for paths in image_paths]


    def load_image(image_paths: Tuple[str, str, str, str]) -> np.ndarray:
        """Load and stack the four channel images."""
        channels = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_paths]
        return np.stack(channels, axis=-1)

    def load_mask(mask_path: str) -> np.ndarray:
        """Load the mask image."""
        return cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = ImageDataset.load_image(self.image_paths[idx])
        filename = self.filenames[idx]
        if self.is_training:
            mask = ImageDataset.load_mask(self.mask_paths[idx])
            return image, mask
        return image, filename

class DeepLabV3DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset: ImageDataset, transformer: Optional[DataTransformer], batch_size: int, is_training: bool = False, shuffle: bool = True):
        """
        Initialize the data generator.
        
        :param dataset: ImageDataset instance
        :param transformer: DataTransformer instance
        :param batch_size: Number of samples per batch
        :param is_training: Whether this generator is for training data (applies augmentation)
        :param shuffle: Whether to shuffle the data after each epoch
        """
        self.dataset = dataset
        self.transformer = transformer
        self.batch_size = batch_size
        self.is_training = is_training
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = []
        if self.is_training:
            y = []
        else: 
            filenames = []
        for i in batch_indexes:
            try:
                if self.is_training:
                    image, mask = self.dataset[i]
                    image, mask = self.transformer.transform_and_augment(image, mask, apply_augmentation=self.is_training)
                    X.append(image)
                    y.append(mask)
                else:
                    image, filename = self.dataset[i]
                    image = self.transformer.transform_image(image)
                    X.append(image)
                    filenames.append(filename)
            except Exception as e:
                print(f"Error processing image at index {i}: {str(e)}")
                continue
        
        if len(X) == 0:
            raise ValueError(f"No valid images found in batch starting at index {index * self.batch_size}")
        
        if self.is_training:
            return np.array(X), np.array(y)
        return np.array(X), filenames

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class CSVLoader:
    def __init__(self, path: str):
        """
        Initialize the CSV loader.
        
        :param path: Path to the CSV file
        """
        self.path = path

    def load_dataframe(self) -> pd.DataFrame:
        """Load the CSV file and return as a DataFrame."""
        return pd.read_csv(self.path)
    
    def load_images_path(self, directory_name: str, is_training: bool = True) -> Tuple[List[List[str]], List[str]]:
        list_img = []
        list_msk = [] if is_training else None
        list_names = self.load_dataframe()

        for filename in tqdm(list_names['name'], miniters=1000):
            channels = ['red', 'green', 'blue', 'nir']
            dir_type_name = "train" if is_training else "test"
            fl_img = [f"{directory_name}{dir_type_name}_{channel}/{channel}_{filename}.TIF" for channel in channels]
            
            list_img.append(fl_img)
            
            if is_training:
                nmask = f"gt_{filename}"
                fl_msk = f"{directory_name}/train_gt/{nmask}.TIF"
                list_msk.append(fl_msk)

        return list_img, list_msk
