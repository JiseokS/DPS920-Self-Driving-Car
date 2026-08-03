# Member 2 - CNN Training

This part is for training the CNN model.

The model takes the road image and gives one steering angle.

```text
image -> CNN model -> steering angle
```

This part does not run the simulator. The simulator test is done after the model is saved.

## Files

These are the main files for this part:

```text
src/model.py
src/main.py
src/evaluate.py
```

`src/model.py` builds the CNN model.

`src/main.py` runs the training.

`src/evaluate.py` saves the training graph.

## Model Input And Output

The model uses the same input from the data/preprocessing part.

```text
Input shape: (66, 200, 3)
Pixel range: 0 to 1
Output: one steering angle
```

Do not add another normalization layer.

The image is already normalized in preprocessing.

## Dataset Folder

The dataset should have `driving_log.csv` and `IMG` folder.

For our data, we have forward and reverse folders.

```text
dataset/
  forward/
    driving_log.csv
    IMG/

  reverse/
    driving_log.csv
    IMG/
```

## Environment

Use the professor/course environment if possible.

The project should match the course `package_list.txt`.

Main versions:

```text
Python 3.8.12
TensorFlow 2.3.0
NumPy 1.21.2
Pandas 1.2.4
Scikit-learn 0.24.2
```

Do not install the newest package versions unless the team agrees.

The `package_list.txt` file is the version reference.

## Train The Model

Before running the command, make sure the dataset is downloaded and extracted on your computer.

The code reads `driving_log.csv` and the images from your local folder.

Make sure the path points to the folder that has `driving_log.csv` and `IMG`.

Run this from the project folder after the environment is active.

```powershell
python -m src.main --dataset-dir "path\to\dataset\forward" "path\to\dataset\reverse" --epochs 10 --batch-size 32
```

Example with my local path:

```powershell
python -m src.main --dataset-dir "C:\Users\John Paul\Downloads\DPS920\project\dataset (1)\dataset\forward" "C:\Users\John Paul\Downloads\DPS920\project\dataset (1)\dataset\reverse" --epochs 10 --batch-size 32
```

## Output Files

After training, the model is saved here:

```text
outputs/models/steering_model.h5
```

The training graph is saved here:

```text
outputs/plots/training_history.png
```

The model file is ignored by Git, so it may need to be shared another way.

## For Simulator Testing

Use this model file:

```text
outputs/models/steering_model.h5
```

Make sure this model file is on your computer before testing.

Before prediction, use the same `preprocess_image()` function from the project.

The model output is one steering angle.

The simulator code should send that steering angle back to the car.

The testing code needs to do this:

```text
load saved model
get image from simulator
preprocess the image
predict steering angle
send steering angle back to simulator
```

## Basic Model Use

For testing or simulator code, load the model like this:

```python
from tensorflow.keras.models import load_model

model = load_model("outputs/models/steering_model.h5")
```

The image should use the same preprocessing function before prediction.

```python
from src.data_preprocessing import preprocess_image

processedImage = preprocess_image(image)
steeringAngle = model.predict(processedImage[None, :, :, :])[0][0]
```

`steeringAngle` is the number that the simulator should use for steering.

If the simulator code uses a different model path, change the path in `load_model()`.

If the simulator image is not RGB, convert it to RGB first before using `preprocess_image()`.

## Notes

Training uses both forward and reverse data.

Training uses balanced training data from the preprocessing part.

Validation data is not balanced.

This code only trains and saves the model. It does not connect to the simulator.

