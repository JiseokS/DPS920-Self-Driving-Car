import numpy as np

from src.data_preprocessing import load_image, preprocess_image


def batch_generator(data, batch_size=32, is_training=True):
    if data.empty:
        raise ValueError("Cannot create batches from an empty dataset.")

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    while True:
        # Shuffle training data at the beginning of each epoch.
        if is_training:
            current_data = data.sample(frac=1).reset_index(drop=True)
        else:
            current_data = data.reset_index(drop=True)

        for start_index in range(0, len(current_data), batch_size):
            batch_data = current_data.iloc[
                start_index : start_index + batch_size
            ]

            images = []
            steering_angles = []

            for row in batch_data.itertuples(index=False):
                image = load_image(row.image_path)
                image = preprocess_image(image)

                images.append(image)
                steering_angles.append(row.steering)

            yield (
                np.array(images, dtype=np.float32),
                np.array(steering_angles, dtype=np.float32),
            )