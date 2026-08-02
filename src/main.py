import argparse
import math
from pathlib import Path

from src.data_preprocessing import ( load_driving_data, split_driving_data, balance_steering_data,)
from src.data_generator import batch_generator
from src.evaluate import plotTrainingHistory
from src.model import buildSteeringNet

EPOCHS = 10
BATCH_SIZE = 32
MODEL_PATH = Path("outputs/models/steering_model.keras")


def getUserOptions():
    """
    Read the values from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    return parser.parse_args()


def main():
    """
    Run the training code.
    """
    userOptions = getUserOptions()

    drivingData = load_driving_data(userOptions.dataset_dir)
    trainData, validationData = split_driving_data(drivingData)

    balancedTrainData = balance_steering_data(trainData)

    trainBatch = batch_generator(
        balancedTrainData,
        batch_size=userOptions.batch_size,
        is_training=True,
    )

    validationBatch = batch_generator(
        validationData,
        batch_size=userOptions.batch_size,
        is_training=False,
    )

    trainSteps = math.ceil(len(balancedTrainData) / userOptions.batch_size)
    validationSteps = math.ceil(len(validationData) / userOptions.batch_size)

    steeringNet = buildSteeringNet()

    trainingHistory = steeringNet.fit(
        trainBatch,
        steps_per_epoch=trainSteps,
        validation_data=validationBatch,
        validation_steps=validationSteps,
        epochs=userOptions.epochs,
    )

    plotTrainingHistory(trainingHistory)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    steeringNet.save(MODEL_PATH)

    print(f"saved model: {MODEL_PATH}")

    return trainingHistory


if __name__ == "__main__":
    main()