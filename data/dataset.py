import numpy as np
import cv2 
import tensorflow as tf
from typing import List, Tuple, Optional
from data.augmentation import DataTransformer
import pandas as pd
from tqdm.auto import tqdm
from utils.utils import extract_filename
from utils.config import Config

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
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        return mask

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
    def __init__(self, dataset: ImageDataset, transformer: Optional[DataTransformer], batch_size: int = Config.BATCH_SIZE, is_training: bool = False, shuffle: bool = True, **kwargs):
        """
        Initialize the data generator.
        
        :param dataset: ImageDataset instance
        :param transformer: DataTransformer instance
        :param batch_size: Number of samples per batch
        :param is_training: Whether this generator is for training data (applies augmentation)
        :param shuffle: Whether to shuffle the data after each epoch
        """
        super().__init__()
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
        X, y, filenames = [], [], []

        for i in batch_indexes:
            try:
                if self.is_training:
                    image, mask = self.dataset[i]
                    if self.transformer:
                        print
                        image, mask = self.transformer.transform_and_augment(image, mask, apply_augmentation=self.is_training)
                    X.append(image)
                    y.append(mask)
                else:
                    image, filename = self.dataset[i]
                    if self.transformer:
                        image = self.transformer.transform_image(image)
                    X.append(image)
                    filenames.append(filename)
            except Exception as e:
                print(f"Error processing image at index {i}: {str(e)}")
                continue

        if self.is_training:
            return np.array(X), np.array(y)
        else:
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
        self.dataframe = self.load_dataframe()

    def load_dataframe(self) -> pd.DataFrame:
        """Load the CSV file and return as a DataFrame."""
        return pd.read_csv(self.path)
    
    def load_images_path(self, directory_name: str, is_training: bool = True) -> Tuple[List[List[str]], List[str]]:
        image_paths = []
        mask_paths = [] if is_training else None

        for filename in tqdm(self.dataframe['name'], miniters=1000):
            image_file_paths = [
                f"{directory_name}{Config.DIR_TYPE_NAME[is_training]}_{channel}/{channel}_{filename}{Config.IMAGE_EXTENSION}"
                for channel in Config.CHANNELS
            ]
            
            image_paths.append(image_file_paths)
            
            if is_training:
                mask_filename = f"{Config.GT_PREFIX}{filename}"
                mask_file_path = f"{directory_name}/{Config.TRAIN_GT_DIR}/{mask_filename}{Config.MASK_EXTENSION}"
                mask_paths.append(mask_file_path)

        return image_paths, mask_paths
