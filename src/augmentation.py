from typing import Tuple

import cv2
import numpy as np


def flip_image(
    image: np.ndarray,
    steering: float,
) -> Tuple[np.ndarray, float]:
    """Flip an image horizontally and reverse its steering angle."""

    flipped_image = cv2.flip(image, 1)
    flipped_steering = -steering

    return flipped_image, flipped_steering


def adjust_brightness(image: np.ndarray) -> np.ndarray:
    """Apply a random brightness adjustment."""

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness_factor = np.random.uniform(0.6, 1.4)

    value_channel = hsv_image[:, :, 2].astype(np.float32)
    value_channel *= brightness_factor
    hsv_image[:, :, 2] = np.clip(value_channel, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def pan_image(image: np.ndarray) -> np.ndarray:
    """Move an image horizontally and vertically by a small amount."""

    height, width = image.shape[:2]

    translation_x = np.random.uniform(-0.1, 0.1) * width
    translation_y = np.random.uniform(-0.1, 0.1) * height

    transformation = np.float32(
        [
            [1, 0, translation_x],
            [0, 1, translation_y],
        ]
    )

    return cv2.warpAffine(
        image,
        transformation,
        (width, height),
        borderMode=cv2.BORDER_REFLECT_101,
    )


def zoom_image(image: np.ndarray) -> np.ndarray:
    """Zoom into an image by a small random amount."""

    height, width = image.shape[:2]
    zoom_factor = np.random.uniform(1.0, 1.2)

    transformation = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        0,
        zoom_factor,
    )

    return cv2.warpAffine(
        image,
        transformation,
        (width, height),
        borderMode=cv2.BORDER_REFLECT_101,
    )


def rotate_image(image: np.ndarray) -> np.ndarray:
    """Rotate an image by a small random angle."""

    height, width = image.shape[:2]
    rotation_angle = np.random.uniform(-5.0, 5.0)

    transformation = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        rotation_angle,
        1.0,
    )

    return cv2.warpAffine(
        image,
        transformation,
        (width, height),
        borderMode=cv2.BORDER_REFLECT_101,
    )


def augment_image(
    image: np.ndarray,
    steering: float,
) -> Tuple[np.ndarray, float]:
    """Apply random augmentation to one training sample."""

    if np.random.random() < 0.5:
        image, steering = flip_image(image, steering)

    if np.random.random() < 0.5:
        image = adjust_brightness(image)

    if np.random.random() < 0.3:
        image = pan_image(image)

    if np.random.random() < 0.3:
        image = zoom_image(image)

    if np.random.random() < 0.3:
        image = rotate_image(image)

    return image, steering