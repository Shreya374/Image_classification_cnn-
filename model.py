"""
CNN architecture — a VGG-inspired classifier built in Keras.

Three Conv->Pool blocks of increasing depth, followed by a dense head with
dropout. Small enough to train quickly on a custom dataset, deep enough to
learn meaningful spatial features.
"""
from tensorflow.keras import layers, models, regularizers
from src.config import IMG_SIZE, NUM_CLASSES


def build_cnn(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE) -> models.Model:
    """Build and compile-ready CNN model.

    Returns an uncompiled Keras Model — caller is responsible for `model.compile(...)`.
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Dense head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="vgg_style_cnn")
    return model


if __name__ == "__main__":
    # Quick sanity check
    m = build_cnn()
    m.summary()
