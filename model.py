import tensorflow as tf


class DeepLabV3Plus(tf.keras.Model):
    def __init__(self, backbone: tf.keras.Model, num_classes: int):
        """
        Initialize the DeepLabV3+ model.
        
        :param backbone: Pre-trained backbone model
        :param num_classes: Number of classes in the dataset
        """
        super(DeepLabV3Plus, self).__init__()
        self.backbone = backbone
        self.aspp = ASPP(backbone.output)
        self.decoder = Decoder(num_classes)
        self.logits = tf.keras.layers.Conv2D(num_classes, 1, activation=None)

    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)
        x = self.aspp(x, training=training)
        x = self.decoder(x, training=training)
        x = self.logits(x)
        return x