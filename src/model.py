from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam


INPUT_SHAPE = (66, 200, 3)
LEARNING_RATE = 0.001


def buildSteeringNet(cameraShape: tuple[int, int, int] = INPUT_SHAPE, adamSpeed: float = LEARNING_RATE,) -> Sequential:
    """
    Build the CNN model for steering angle.

    Args:
        cameraShape: Input image shape.
        adamSpeed: Adam learning rate.

    Returns:
        The compiled model.
    """

    steeringNet = Sequential([
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=cameraShape),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu"),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dense(50, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(1),
    ])

    steeringNet.compile(
        optimizer=Adam(learning_rate=adamSpeed),
        loss="mse",
        metrics=["mae"],
    )

    return steeringNet