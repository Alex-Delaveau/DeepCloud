import tensorflow as tf
from tensorflow.keras import layers, Model

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = f'expanded_conv_{block_id}_'

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = layers.Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                               use_bias=False, padding='same', dilation_rate=(rate, rate),
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = layers.Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if skip_connection:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x

def DeepLabv3Plus(input_shape=(512, 512, 3), alpha=1.0):
    img_input = layers.Input(shape=input_shape)

    # MobileNetV2 feature extractor
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, name='Conv')(img_input)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = layers.Activation(tf.nn.relu6, name='Conv_Relu6')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5, skip_connection=True)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=12, skip_connection=True)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2, expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4, expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4, expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4, expansion=6, block_id=16, skip_connection=False)

    # ASPP
    shape_before = tf.keras.backend.int_shape(x)
    b4 = layers.GlobalAveragePooling2D()(x)
    b4 = layers.Reshape((1, 1, shape_before[3]))(b4)
    b4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = layers.Activation(tf.nn.relu)(b4)
    b4 = layers.UpSampling2D(size=(shape_before[1], shape_before[2]), interpolation='bilinear')(b4)

    b0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = layers.Activation(tf.nn.relu, name='aspp0_activation')(b0)

    x = layers.Concatenate()([b4, b0])

    x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = layers.Activation(tf.nn.relu)(x)
    x = layers.Dropout(0.1)(x)

    # DeepLab v.3+ decoder
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    
    # Final convolution
    x = layers.Conv2D(1, (1, 1), padding='same', name='final_conv')(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    # Ensure output size matches input size
    x = layers.Resizing(input_shape[0], input_shape[1], interpolation='bilinear')(x)

    # Apply sigmoid activation for binary segmentation
    x = layers.Activation('sigmoid', name='output_layer')(x)

    model = Model(img_input, x, name='deeplabv3plus_mobilenetv2_binary')

    return model

# Example usage
if __name__ == "__main__":
    model = DeepLabv3Plus(input_shape=(512, 512, 3), alpha=1.0)
    model.summary()