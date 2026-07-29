"""Load and prepare Udacity simulator driving data."""

from pathlib import Path
from typing import Union

import pandas as pd


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
    # Load center-camera paths and steering angles from driving_log.csv.

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
    # If a header row exists, it becomes NaN and is removed safely.
    data["steering"] = pd.to_numeric(data["steering"], errors="coerce")
    data = data.dropna(subset=["center", "steering"]).copy()

    def rebuild_image_path(recorded_path: object) -> str:
        cleaned_path = str(recorded_path).strip().replace("\\", "/")
        filename = cleaned_path.split("/")[-1]
        return str(image_dir / filename)

    # Keep only the filename so the dataset works on different computers.
    data["image_path"] = data["center"].map(rebuild_image_path)

    valid_mask = data["image_path"].map(lambda path: Path(path).is_file())
    missing_count = int((~valid_mask).sum())

    if missing_count:
        print(f"[WARNING] Ignoring {missing_count} missing center images.")

    result = data.loc[valid_mask, ["image_path", "steering"]]
    result = result.reset_index(drop=True)

    if result.empty:
        raise ValueError("No valid center-camera samples were found.")

    print(f"[INFO] Loaded {len(result)} valid driving samples.")
    return result