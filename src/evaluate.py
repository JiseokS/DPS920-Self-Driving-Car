from pathlib import Path

import matplotlib.pyplot as plt


PLOT_PATH = Path("outputs/plots/training_history.png")


def plotTrainingHistory(trainingHistory, plotPath: Path = PLOT_PATH):
    """
    Plot the training result.
    """

    historyData = trainingHistory.history

    lossValues = historyData.get("loss", [])
    valLossValues = historyData.get("val_loss", [])
    maeValues = historyData.get("mae", [])
    valMaeValues = historyData.get("val_mae", [])

    plotPath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(lossValues, label="Training Loss")
    plt.plot(valLossValues, label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(maeValues, label="Training MAE")
    plt.plot(valMaeValues, label="Validation MAE")
    plt.title("Model MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plotPath)
    plt.close()

    print(f"saved graph: {plotPath}")