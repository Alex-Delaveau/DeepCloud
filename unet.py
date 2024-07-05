import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(inputs, filters):
    x = layers.Conv2D(filters, 3, padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.ReLU()(x)
    return x

def create_unet_model(input_shape=(384, 384, 4), out_channels=1):
    inputs = layers.Input(shape=input_shape)

    # Downsampling
    conv1 = conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    bottleneck = conv_block(pool2, 256)

    # Upsampling
    up4 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(bottleneck)
    up4 = layers.concatenate([up4, conv2])
    conv4 = conv_block(up4, 128)

    up5 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(conv4)
    up5 = layers.concatenate([up5, conv1])
    conv5 = conv_block(up5, 64)

    # Output layer
    outputs = layers.Conv2D(out_channels, 1, activation='sigmoid')(conv5)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model