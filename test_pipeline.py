import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data_preprocessing import (
    load_driving_data,
    load_image,
    preprocess_image,
    split_driving_data,
)
from src.data_generator import batch_generator

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", required=True)
args = parser.parse_args()

data = load_driving_data(args.dataset_dir)

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

train_data, validation_data = split_driving_data(data)

train_generator = batch_generator(
    train_data,
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