# Member 1 – Data and Preprocessing

## Files

### `src/data_preprocessing.py`

Contains functions for:

- Loading center-camera paths and steering angles from `driving_log.csv`
- Rebuilding image paths for the current computer
- Removing rows with missing images or invalid steering values
- Splitting data into training and validation sets
- Loading simulator images
- Cropping, converting to YUV, blurring, resizing, and normalizing images

Processed image output:

```text
Shape: (66, 200, 3)
Data type: float32
Pixel range: 0–1
```

### `src/data_generator.py`

Contains the batch generator used to supply processed images and steering angles.

```text
Images:          (batch_size, 66, 200, 3)
Steering angles: (batch_size,)
```

Training data is shuffled before each epoch. Validation data keeps its original order.

### `test_pipeline.py`

Tests the current data pipeline, including:

- Data loading
- Image preprocessing
- Train/validation split
- Training and validation batches
- Image shape and pixel range
- Invalid numerical values

It also saves a preprocessing comparison image to:

```text
outputs/plots/preprocessing_comparison.png
```

## Testing

Activate the project environment:

```powershell
conda activate dps920-final
```

Run the test from the project root:

```powershell
python test_pipeline.py --dataset-dir "..\test_data"
```

A successful test ends with:

```text
Batch pipeline test passed.
```