import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters, kernel_size=3, strides=1, use_bn=True):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    return x

def inverted_residual_block(x, expand, squeeze, strides):
    m = conv_block(x, expand, 1)
    m = layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    m = layers.ReLU(6.)(m)
    m = layers.Conv2D(squeeze, 1, padding='same', use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    
    if strides == 1 and x.shape[-1] == squeeze:
        return layers.Add()([m, x])
    return m

def encoder_block(x, filters, strides):
    x = inverted_residual_block(x, filters * 4, filters, strides)
    return x

def decoder_block(x, skip, filters):
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_lightweight_unet(input_shape, out_channels):
    inputs = layers.Input(input_shape)
    
    # Encoder (MobileNetV3Small-inspired)
    x = conv_block(inputs, 16, strides=2)  # 96x96
    s1 = encoder_block(x, 16, 1)           # 96x96
    s2 = encoder_block(s1, 24, 2)          # 48x48
    s3 = encoder_block(s2, 40, 2)          # 24x24
    s4 = encoder_block(s3, 80, 2)          # 12x12
    x = encoder_block(s4, 112, 1)          # 12x12
    x = encoder_block(x, 160, 2)           # 6x6
    
    # Bridge
    b = conv_block(x, 320, 1)              # 6x6
    
    # Decoder
    x = decoder_block(b, s4, 256)          # 12x12
    x = decoder_block(x, s3, 128)          # 24x24
    x = decoder_block(x, s2, 64)           # 48x48
    x = decoder_block(x, s1, 32)           # 96x96
    
    # Output
    x = layers.UpSampling2D((2, 2))(x)     # 192x192
    outputs = layers.Conv2D(out_channels, 1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    return model

def build_super_lightweight_unet(input_shape, out_channels):
    inputs = layers.Input(input_shape)
    
    # Encoder (Simplified MobileNetV3Small-inspired)
    x = conv_block(inputs, 8, strides=2)   # 96x96
    s1 = encoder_block(x, 8, 1)            # 96x96
    s2 = encoder_block(s1, 16, 2)          # 48x48
    s3 = encoder_block(s2, 24, 2)          # 24x24
    s4 = encoder_block(s3, 32, 2)          # 12x12
    
    # Bridge
    b = conv_block(s4, 64, 1)              # 12x12
    
    # Decoder
    x = decoder_block(b, s3, 24)           # 24x24
    x = decoder_block(x, s2, 16)           # 48x48
    x = decoder_block(x, s1, 8)            # 96x96
    
    # Output
    x = layers.UpSampling2D((2, 2))(x)     # 192x192
    outputs = layers.Conv2D(out_channels, 1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    return model
