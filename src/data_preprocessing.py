from pathlib import Path
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LOG_COLUMNS = [
    "center",
    "left",
    "right",
    "steering",
    "throttle",
    "brake",
    "speed",
]


def load_driving_data(dataset_dir: Union[str, Path]) -> pd.DataFrame:
    """Load center-camera paths and steering angles from driving_log.csv."""

    dataset_path = Path(dataset_dir).expanduser().resolve()
    csv_path = dataset_path / "driving_log.csv"
    image_dir = dataset_path / "IMG"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Driving log not found: {csv_path}")

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    data = pd.read_csv(
        csv_path,
        header=None,
        names=LOG_COLUMNS,
        skipinitialspace=True,
    )

    # Convert steering values to numbers.
    data["steering"] = pd.to_numeric(data["steering"], errors="coerce")
    data = data.dropna(subset=["center", "steering"]).copy()

    def rebuild_image_path(recorded_path: object) -> str:
        cleaned_path = str(recorded_path).strip().replace("\\", "/")
        filename = cleaned_path.split("/")[-1]
        return str(image_dir / filename)

    # Rebuild paths so the dataset works on different computers.
    data["image_path"] = data["center"].map(rebuild_image_path)

    valid_mask = data["image_path"].map(
        lambda image_path: Path(image_path).is_file()
    )
    missing_count = int((~valid_mask).sum())

    if missing_count:
        print(f"[WARNING] Ignoring {missing_count} missing center images.")

    result = data.loc[valid_mask, ["image_path", "steering"]]
    result = result.reset_index(drop=True)

    if result.empty:
        raise ValueError("No valid center-camera samples were found.")

    print(f"[INFO] Loaded {len(result)} valid driving samples.")
    return result


def split_driving_data(
    data: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split driving data into training and validation sets."""

    train_data, validation_data = train_test_split(
        data,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )

    train_data = train_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)

    print(f"[INFO] Training samples: {len(train_data)}")
    print(f"[INFO] Validation samples: {len(validation_data)}")

    return train_data, validation_data
    
    
def plot_steering_distribution(
    data: pd.DataFrame,
    output_path: Union[str, Path],
    bins: int = 25,
    title: str = "Steering Angle Distribution",
) -> None:
    """Save a histogram of the steering-angle distribution."""

    if data.empty:
        raise ValueError("Cannot plot an empty dataset.")

    if bins <= 0:
        raise ValueError("The number of bins must be greater than zero.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.hist(data["steering"], bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel("Steering Angle")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"[INFO] Saved steering histogram: {output_path}")


def balance_steering_data(
    data: pd.DataFrame,
    bins: int = 25,
    max_samples_per_bin: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Limit the number of training samples in each steering-angle bin."""

    if data.empty:
        raise ValueError("Cannot balance an empty dataset.")

    if bins <= 0:
        raise ValueError("The number of bins must be greater than zero.")

    if max_samples_per_bin <= 0:
        raise ValueError(
            "The maximum samples per bin must be greater than zero."
        )

    steering_values = data["steering"].to_numpy()

    # Divide the complete steering range into equal-width bins.
    bin_edges = np.linspace(
        steering_values.min(),
        steering_values.max(),
        bins + 1,
    )

    # Assign every sample to one of the steering bins.
    bin_indices = np.digitize(
        steering_values,
        bin_edges[1:-1],
    )

    balanced_groups = []

    for bin_index in range(bins):
        group_indices = np.flatnonzero(bin_indices == bin_index)
        group = data.iloc[group_indices]

        # Keep all samples from small bins and reduce only large bins.
        if len(group) > max_samples_per_bin:
            group = group.sample(
                n=max_samples_per_bin,
                random_state=random_state + bin_index,
            )

        balanced_groups.append(group)

    balanced_data = pd.concat(
        balanced_groups,
        ignore_index=True,
    )

    # Shuffle the remaining samples after balancing.
    balanced_data = balanced_data.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    print(
        f"[INFO] Balanced training samples: "
        f"{len(data)} -> {len(balanced_data)}"
    )

    return balanced_data


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load an image from a file path and convert it to RGB format."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert BGR to RGB to match the simulator image format.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Crop, convert, blur, resize, and normalize a simulator image."""

    image = image[60:135, :, :]

    if image.size == 0:
        raise ValueError("Image is too small for preprocessing.")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image.astype(np.float32) / 255.0

    return image