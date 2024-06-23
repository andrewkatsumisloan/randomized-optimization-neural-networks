import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "./plots3"


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(
            2.0 / input_size
        )
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(
            2.0 / hidden_size
        )
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def loss(self, X, y):
        epsilon = 1e-15
        y_pred = np.clip(self.forward(X), epsilon, 1 - epsilon)
        return np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))

    def accuracy(self, X, y):
        y_pred = self.forward(X)
        y_pred_class = (y_pred > 0.5).astype(int)
        return np.mean(y_pred_class == y)

    def get_params(self):
        return np.concatenate(
            [self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()]
        )

    def set_params(self, params):
        w1_end = self.input_size * self.hidden_size
        b1_end = w1_end + self.hidden_size
        w2_end = b1_end + self.hidden_size * self.output_size

        self.W1 = params[:w1_end].reshape(self.input_size, self.hidden_size)
        self.b1 = params[w1_end:b1_end].reshape(1, self.hidden_size)
        self.W2 = params[b1_end:w2_end].reshape(self.hidden_size, self.output_size)
        self.b2 = params[w2_end:].reshape(1, self.output_size)


# Load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def plot_performance_multiple_lines(losses, accuracies, title, filename, params_list):
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (loss, accuracy) in enumerate(zip(losses, accuracies)):
        param_str = ", ".join(
            [f"{k}={v}" for k, v in params_list[i].items() if k != "hidden_size"]
        )
        ax.plot(accuracy, label=param_str)

    ax.set_title(f"{title} - Accuracy vs Iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def randomized_hill_climbing(nn, X, y, step_size, iterations, run_name):
    current_params = nn.get_params()
    current_loss = nn.loss(X, y)
    best_params = current_params
    best_loss = current_loss

    losses = []
    accuracies = []

    for i in range(iterations):
        neighbor_params = current_params + np.random.normal(
            0, step_size, size=current_params.shape
        )
        nn.set_params(neighbor_params)
        neighbor_loss = nn.loss(X, y)

        if neighbor_loss < current_loss:
            current_params = neighbor_params
            current_loss = neighbor_loss

            if current_loss < best_loss:
                best_params = current_params
                best_loss = current_loss

        losses.append(current_loss)
        accuracies.append(nn.accuracy(X, y))

    nn.set_params(best_params)
    return losses, accuracies


def run_neural_network(file_path, hidden_size, optimization_method, run_name, **kwargs):
    X_train, X_test, y_train, y_test = load_data(file_path)
    input_size = X_train.shape[1]
    output_size = 1

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    if optimization_method == "randomized_hill_climbing":
        losses, accuracies = randomized_hill_climbing(
            nn, X_train, y_train, run_name=run_name, **kwargs
        )
    else:
        raise ValueError(
            "Invalid optimization method. Choose 'randomized_hill_climbing'."
        )

    return losses, accuracies


def compare_hyperparameters(file_path, optimization_method, param_ranges):
    all_losses = []
    all_accuracies = []

    for i, params in enumerate(param_ranges):
        run_name = "_".join(
            [f"{k}={v}" for k, v in params.items() if k != "hidden_size"]
        )
        hidden_size = params.pop("hidden_size")
        losses, accuracies = run_neural_network(
            file_path,
            hidden_size=hidden_size,
            optimization_method=optimization_method,
            run_name=run_name,
            **params,
        )
        all_losses.append(losses)
        all_accuracies.append(accuracies)

    plot_performance_multiple_lines(
        all_losses,
        all_accuracies,
        optimization_method,
        f"{optimization_method}_accuracy_comparison.png",
        param_ranges,
    )


# Create a directory for saving plots
os.makedirs(PLOTS_DIR, exist_ok=True)

# Example usage
file_path = "./dataset/diabetes_balanced.csv"

# Randomized Hill Climbing
rhc_params = [
    {"hidden_size": 30, "step_size": 0.1, "iterations": 1000},
    {"hidden_size": 30, "step_size": 0.05, "iterations": 2000},
    {"hidden_size": 30, "step_size": 0.025, "iterations": 4000},
]
compare_hyperparameters(file_path, "randomized_hill_climbing", rhc_params)

print("All plots have been saved in the 'plots' directory.")
