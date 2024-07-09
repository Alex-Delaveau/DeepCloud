from sklearn.model_selection import train_test_split


from data.dataset import ImageDataset, DeepLabV3DataGenerator, CSVLoader
from data.augmentation import DataTransformer
from models.unet import create_unet_model
from models.cloudnet import create_cloudnet_model
from models.callbacks import get_callbacks
from utils.config import Config
from models.deeplab import DeepLabv3Plus

def create_train_val_generators(val_ratio: float = 0.2,
                                random_state: int = 42):
    """
    Create train and validation generators.
    
    :param img_rows: Number of rows in the output image
    :param img_cols: Number of columns in the output image
    :param batch_size: Number of samples per batch
    :param val_ratio: Ratio of validation data
    :param max_possible_input_value: Maximum possible pixel value in the input images
    :param random_state: Random state for reproducibility
    :return: Tuple of (train_generator, val_generator)
    """

    csv_loader = CSVLoader(Config.TRAIN_CSV)
    image_paths, mask_paths = csv_loader.load_images_path(Config.TRAIN_PATH)

    # create subset
    image_paths = image_paths[:100]
    mask_paths = mask_paths[:100]


    # Split the data
    train_images_path, val_images_path, train_masks_path, val_masks_path = train_test_split(
        image_paths, mask_paths, test_size=val_ratio, random_state=random_state)

    # Create datasets
    train_dataset = ImageDataset(train_images_path, train_masks_path)
    val_dataset = ImageDataset(val_images_path, val_masks_path)

    # Create transformer
    transformer = DataTransformer()

    # Create generators
    train_generator = DeepLabV3DataGenerator(train_dataset, transformer, shuffle=True, is_training=True, 
                                            workers=4,
                                            use_multiprocessing=True,
                                            max_queue_size=10)
    val_generator = DeepLabV3DataGenerator(val_dataset, transformer, shuffle=False, is_training=True,
                                            workers=4,
                                            use_multiprocessing=True,
                                            max_queue_size=10)

    return train_generator, val_generator


def train_unet(train_generator, val_generator, input_shape = Config.MODEL_INPUT_SHAPE , max_num_epochs = Config.MAX_EPOCHS):
    callbacks = get_callbacks(model_name='unet')
    unet = create_unet_model(input_shape, 1)
    unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    unet.fit(
        train_generator,
        validation_data=val_generator,
        epochs=max_num_epochs,
        callbacks=callbacks
    )

def train_cloudnet(train_generator, val_generator, input_shape = Config.MODEL_INPUT_SHAPE , max_num_epochs = Config.MAX_EPOCHS):
    callbacks = get_callbacks(model_name='cloud-net')
    cloudnet = create_cloudnet_model(input_shape=input_shape, out_channels=1)
    cloudnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cloudnet.fit(
        train_generator,
        validation_data=val_generator,
        epochs=max_num_epochs,
        callbacks=callbacks
    )

def train_deeplab(train_generator, val_generator, input_shape = Config.MODEL_INPUT_SHAPE , max_num_epochs = Config.MAX_EPOCHS):
    callbacks = get_callbacks(model_name='deeplab')
    deeplab = DeepLabv3Plus(input_shape=input_shape)
    deeplab.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    deeplab.fit(
        train_generator,
        validation_data=val_generator,
        epochs=max_num_epochs,
        callbacks=callbacks
    )

def main():
    train_generator, val_generator = create_train_val_generators()
    # Train the model
    # train_unet(train_generator, val_generator)
    train_deeplab(train_generator, val_generator)

# Call main
if __name__ == "__main__":
    main()