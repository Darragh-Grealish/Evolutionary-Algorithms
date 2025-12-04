import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

def plot_generation_times(generation_times):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(generation_times)), generation_times, marker='o')
    plt.title('Generation Times Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_results(y_test, y_pred, show_mae=True):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)


    # MAE
    if show_mae:
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error: {mae:,.2f}")


    # actual & predicted scatter graph
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal: y = ŷ')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs. Predicted House Prices')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # residual graph
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 4))
    plt.scatter(range(len(residuals)), residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Error (Actual - Predicted)')
    plt.title('Residuals (Prediction Error) per Sample')
    plt.tight_layout()
    plt.show()


