import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data_preprocessing import (
    load_driving_data,
    load_image,
    preprocess_image,
)


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