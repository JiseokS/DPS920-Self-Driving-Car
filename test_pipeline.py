import argparse
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.augmentation import (
    adjust_brightness,
    flip_image,
    pan_image,
    rotate_image,
    zoom_image,
)

from src.data_preprocessing import (
    load_driving_data,
    load_image,
    plot_steering_distribution,
    preprocess_image,
    split_driving_data,
    balance_steering_data,
)
from src.data_generator import batch_generator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-dir",
    required=True,
    nargs="+",
)
args = parser.parse_args()

datasets = [
    load_driving_data(dataset_dir)
    for dataset_dir in args.dataset_dir
]

data = pd.concat(
    datasets,
    ignore_index=True,
)

print(f"[INFO] Combined driving samples: {len(data)}")

original = load_image(data.loc[0, "image_path"])
processed = preprocess_image(original)

# Convert the processed YUV image back to RGB for display.
processed_uint8 = (processed * 255).astype(np.uint8)
processed_rgb = cv2.cvtColor(processed_uint8, cv2.COLOR_YUV2RGB)

output_path = Path("outputs/plots/preprocessing_comparison.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed_rgb)
plt.title("Preprocessed")
plt.axis("off")

plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"Original shape: {original.shape}")
print(f"Processed shape: {processed.shape}")
print(f"Saved comparison: {output_path}")

# Select a sample with a visible steering angle.
turning_samples = data[data["steering"].abs() > 0.05]

if turning_samples.empty:
    augmentation_sample = data.iloc[0]
else:
    augmentation_sample = turning_samples.iloc[0]

augmentation_original = load_image(augmentation_sample["image_path"])
original_steering = float(augmentation_sample["steering"])

# Use a fixed seed so the saved result is reproducible.
np.random.seed(42)

flipped_image, flipped_steering = flip_image(
    augmentation_original,
    original_steering,
)

augmentation_results = [
    ("Original", augmentation_original, original_steering),
    ("Flip", flipped_image, flipped_steering),
    (
        "Brightness",
        adjust_brightness(augmentation_original.copy()),
        original_steering,
    ),
    ("Pan", pan_image(augmentation_original.copy()), original_steering),
    ("Zoom", zoom_image(augmentation_original.copy()), original_steering),
    (
        "Rotation",
        rotate_image(augmentation_original.copy()),
        original_steering,
    ),
]

augmentation_output_path = Path(
    "outputs/plots/augmentation_examples.png"
)

plt.figure(figsize=(12, 7))

for index, (name, image, steering) in enumerate(
    augmentation_results,
    start=1,
):
    assert image.shape == augmentation_original.shape

    plt.subplot(2, 3, index)
    plt.imshow(image)
    plt.title(f"{name}\nSteering: {steering:.3f}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(augmentation_output_path)
plt.close()

print(f"Saved augmentation examples: {augmentation_output_path}")

train_data, validation_data = split_driving_data(data)

plot_steering_distribution(
    train_data,
    "outputs/plots/steering_before_balancing.png",
    bins=25,
    title="Training Steering Distribution Before Balancing",
)

balanced_train_data = balance_steering_data(
    train_data,
    bins=25,
    max_samples_per_bin=200,
    random_state=42,
)

plot_steering_distribution(
    balanced_train_data,
    "outputs/plots/steering_after_balancing.png",
    bins=25,
    title="Training Steering Distribution After Balancing",
)

assert len(balanced_train_data) <= len(train_data)

train_generator = batch_generator(
    balanced_train_data,
    batch_size=32,
    is_training=True,
)

validation_generator = batch_generator(
    validation_data,
    batch_size=32,
    is_training=False,
)

train_images, train_steering = next(train_generator)
validation_images, validation_steering = next(validation_generator)

assert train_images.shape == (32, 66, 200, 3)
assert train_steering.shape == (32,)
assert validation_images.shape == (32, 66, 200, 3)
assert validation_steering.shape == (32,)

assert np.isfinite(train_images).all()
assert np.isfinite(train_steering).all()
assert np.isfinite(validation_images).all()
assert np.isfinite(validation_steering).all()

assert 0.0 <= train_images.min() <= train_images.max() <= 1.0
assert 0.0 <= validation_images.min() <= validation_images.max() <= 1.0

print(f"Train batch images: {train_images.shape}")
print(f"Train batch steering: {train_steering.shape}")
print(f"Validation batch images: {validation_images.shape}")
print(f"Validation batch steering: {validation_steering.shape}")
print("Batch pipeline test passed.")