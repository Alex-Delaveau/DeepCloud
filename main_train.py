import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

from dataset import CSVLoader
from dataset import ImageDataset, DeepLabV3DataGenerator
from augmentation import DataTransformer
from unet import create_unet_model
from cloudnet import create_cloudnet_model

GLOBAL_PATH = '../CloudNet/Full_Cloud/'
TRAIN_PATH = os.path.join(GLOBAL_PATH, 'train/')
TRAIN_CSV = os.path.join(TRAIN_PATH, 'train_set.csv')
SAVE_PATH = 'saved_models/'

def get_callbacks(model_name='unet', monitor='val_loss'):
    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=10,
            verbose=1,
            mode='min',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.1,
            patience=5,
            verbose=1,
            mode='min',
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            filepath=os.path.join(SAVE_PATH,f'best_{model_name}.h5'),
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            mode='min'
        ),
        TensorBoard(
            log_dir=f'./logs/{model_name}',
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=2,
            embeddings_freq=1
        )
    ]
    return callbacks

def create_train_val_generators(img_rows: int,
                                img_cols: int,
                                batch_size: int,
                                val_ratio: float = 0.2,
                                max_possible_input_value: float = 65536,
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

    csv_loader = CSVLoader(TRAIN_CSV)
    image_paths, mask_paths = csv_loader.load_images_path(TRAIN_PATH)

    # Split the data
    train_images_path, val_images_path, train_masks_path, val_masks_path = train_test_split(
        image_paths, mask_paths, test_size=val_ratio, random_state=random_state)

    # Create datasets
    train_dataset = ImageDataset(train_images_path, train_masks_path)
    val_dataset = ImageDataset(val_images_path, val_masks_path)

    # Create transformer
    transformer = DataTransformer(img_rows, img_cols, max_possible_input_value)

    # Create generators
    train_generator = DeepLabV3DataGenerator(train_dataset, transformer, batch_size, shuffle=True, is_training=True)
    val_generator = DeepLabV3DataGenerator(val_dataset, transformer, batch_size, shuffle=False, is_training=True)

    return train_generator, val_generator

def train_unet(input_shape, train_generator, val_generator, max_num_epochs):
    callbacks = get_callbacks(model_name='unet')
    unet = create_unet_model(input_shape=input_shape, out_channels=1)
    unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = unet.fit(
        train_generator,
        validation_data=val_generator,
        epochs=max_num_epochs,
        callbacks=callbacks
    )

def train_cloudnet(input_shape, train_generator, val_generator, max_num_epochs):
    callbacks = get_callbacks(model_name='cloud-net')
    cloudnet = create_cloudnet_model(input_shape=input_shape, out_channels=1)
    cloudnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = cloudnet.fit(
        train_generator,
        validation_data=val_generator,
        epochs=max_num_epochs,
        callbacks=callbacks
    )

def main():
    max_num_epochs = 2000 
    train_generator, val_generator = create_train_val_generators(392, 392, 12, 0.2, 65536, 42)
    
    input_shape = (392, 392, 4)

    # Train the model
    
    train_unet(input_shape, train_generator, val_generator, max_num_epochs)

# Call main
if __name__ == "__main__":
    main()