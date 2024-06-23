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

        # Initialize weights using He initialization for ReLU
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


def plot_performance(losses, accuracies, title, filename, params):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(losses)
    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")

    ax2.plot(accuracies)
    ax2.set_title(f"{title} - Accuracy")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Accuracy")

    # Add hyperparameters as text
    param_text = "\n".join(
        [f"{k}: {v}" for k, v in params.items() if k != "hidden_size"]
    )
    fig.text(0.95, 0.05, param_text, fontsize=8, ha="right", va="bottom")

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
    plot_performance(
        losses,
        accuracies,
        f"Randomized Hill Climbing",
        f"RHC_{run_name}_performance.png",
        {"step_size": step_size, "iterations": iterations},
    )
    return best_loss


def simulated_annealing(
    nn,
    X,
    y,
    initial_temp,
    cooling_rate,
    max_iterations,
    run_name,
    min_temp=1e-6,
    convergence_iterations=50,
    convergence_threshold=1e-6,
    cool_down_type="linear",
):
    current_params = nn.get_params()
    current_loss = nn.loss(X, y)
    best_params = current_params
    best_loss = current_loss
    temp = initial_temp

    losses = []
    accuracies = []
    iterations_without_improvement = 0

    for i in range(max_iterations):
        new_params = current_params + np.random.normal(
            0, 0.1, size=current_params.shape
        )
        nn.set_params(new_params)
        new_loss = nn.loss(X, y)

        if new_loss < current_loss or np.random.random() < np.exp(
            (current_loss - new_loss) / temp
        ):
            current_params = new_params
            current_loss = new_loss

            if current_loss < best_loss - convergence_threshold:
                best_params = current_params
                best_loss = current_loss
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
        else:
            iterations_without_improvement += 1

        # Implement exponential cool down schedule
        if cool_down_type == "exponential":
            temp = initial_temp * (cooling_rate**i)
        else:  # linear cool down (original implementation)
            temp *= cooling_rate

        losses.append(current_loss)
        accuracies.append(nn.accuracy(X, y))

        # Check for convergence
        if iterations_without_improvement >= convergence_iterations or temp < min_temp:
            print(f"Converged after {i + 1} iterations")
            break

    nn.set_params(best_params)
    plot_performance(
        losses,
        accuracies,
        f"Simulated Annealing ({cool_down_type} cool down)",
        f"SA_{run_name}_performance_{cool_down_type}.png",
        {
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
            "max_iterations": max_iterations,
            "min_temp": min_temp,
            "convergence_iterations": convergence_iterations,
            "convergence_threshold": convergence_threshold,
            "cool_down_type": cool_down_type,
        },
    )
    return best_loss


def genetic_algorithm(
    nn,
    X,
    y,
    population_size,
    max_generations,
    mutation_rate,
    run_name,
    convergence_generations=50,
    convergence_threshold=1e-6,
):
    def create_individual():
        return nn.get_params() + np.random.normal(0, 0.1, size=nn.get_params().shape)

    population = [create_individual() for _ in range(population_size)]

    losses = []
    accuracies = []
    best_loss = float("inf")
    generations_without_improvement = 0

    for gen in range(max_generations):
        fitness_scores = []
        for individual in population:
            nn.set_params(individual)
            fitness_scores.append(-nn.loss(X, y))

        parents = []
        for _ in range(population_size):
            tournament = np.random.choice(population_size, 3, replace=False)
            winner = max(tournament, key=lambda i: fitness_scores[i])
            parents.append(population[winner])

        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            crossover_point = np.random.randint(len(parent1))
            child1 = np.concatenate(
                [parent1[:crossover_point], parent2[crossover_point:]]
            )
            child2 = np.concatenate(
                [parent2[:crossover_point], parent1[crossover_point:]]
            )
            new_population.extend([child1, child2])

        for individual in new_population:
            if np.random.random() < mutation_rate:
                individual += np.random.normal(0, 0.1, size=individual.shape)

        population = new_population

        best_individual = max(population, key=lambda ind: -nn.loss(X, y))
        nn.set_params(best_individual)
        current_loss = nn.loss(X, y)
        current_accuracy = nn.accuracy(X, y)
        losses.append(current_loss)
        accuracies.append(current_accuracy)

        # Check for convergence
        if current_loss < best_loss - convergence_threshold:
            best_loss = current_loss
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= convergence_generations:
            print(f"Converged after {gen + 1} generations")
            break

    plot_performance(
        losses,
        accuracies,
        f"Genetic Algorithm",
        f"GA_{run_name}_performance.png",
        {
            "population_size": population_size,
            "max_generations": max_generations,
            "mutation_rate": mutation_rate,
            "convergence_generations": convergence_generations,
            "convergence_threshold": convergence_threshold,
        },
    )
    return nn.loss(X, y)


# Modify run_neural_network function to include RHC
def run_neural_network(file_path, hidden_size, optimization_method, run_name, **kwargs):
    X_train, X_test, y_train, y_test = load_data(file_path)
    input_size = X_train.shape[1]
    output_size = 1

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    if optimization_method == "simulated_annealing":
        best_loss = simulated_annealing(
            nn, X_train, y_train, run_name=run_name, **kwargs
        )
    elif optimization_method == "genetic_algorithm":
        best_loss = genetic_algorithm(nn, X_train, y_train, run_name=run_name, **kwargs)
    elif optimization_method == "randomized_hill_climbing":
        best_loss = randomized_hill_climbing(
            nn, X_train, y_train, run_name=run_name, **kwargs
        )
    else:
        raise ValueError(
            "Invalid optimization method. Choose 'simulated_annealing', 'genetic_algorithm', or 'randomized_hill_climbing'."
        )

    train_loss = nn.loss(X_train, y_train)
    test_loss = nn.loss(X_test, y_test)
    train_accuracy = nn.accuracy(X_train, y_train)
    test_accuracy = nn.accuracy(X_test, y_test)

    print(f"Run: {run_name}")
    print(f"Best training loss: {train_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print()

    return nn, train_loss, test_loss, train_accuracy, test_accuracy


def compare_hyperparameters(file_path, optimization_method, param_ranges):
    results = []
    for i, params in enumerate(param_ranges):
        param_str = "_".join(
            [f"{k}={v}" for k, v in params.items() if k != "hidden_size"]
        )
        run_name = f"{param_str}"
        nn, train_loss, test_loss, train_acc, test_acc = run_neural_network(
            file_path,
            optimization_method=optimization_method,
            run_name=run_name,
            **params,
        )
        results.append((params, train_loss, test_loss, train_acc, test_acc))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.set_title(f"{optimization_method} - Loss vs Hyperparameters")
    ax2.set_title(f"{optimization_method} - Accuracy vs Hyperparameters")

    for i, (params, train_loss, test_loss, train_acc, test_acc) in enumerate(results):
        param_str = ", ".join(
            [f"{k}={v}" for k, v in params.items() if k not in ["hidden_size"]]
        )
        x = i
        ax1.scatter([x, x], [train_loss, test_loss], label=["Train", "Test"])
        ax2.scatter([x, x], [train_acc, test_acc], label=["Train", "Test"])

    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(
        [f"Run {i+1}" for i in range(len(results))], rotation=45, ha="right"
    )
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(
        [f"Run {i+1}" for i in range(len(results))], rotation=45, ha="right"
    )
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    # Add hyperparameters as text
    for i, (params, _, _, _, _) in enumerate(results):
        param_text = "\n".join(
            [f"{k}: {v}" for k, v in params.items() if k != "hidden_size"]
        )
        ax1.annotate(
            param_text,
            (i, 0),
            xytext=(0, -40),
            textcoords="offset points",
            va="top",
            ha="center",
            fontsize=8,
            rotation=90,
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{optimization_method}_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# Create a directory for saving plots
os.makedirs(PLOTS_DIR, exist_ok=True)

# Example usage
file_path = "./dataset/diabetes_balanced.csv"

# Simulated Annealing
sa_params = [
    # {
    #     "hidden_size": 30,
    #     "initial_temp": 150,
    #     "cooling_rate": 0.99,
    #     "max_iterations": 2000,
    #     "min_temp": 1e-6,
    #     "convergence_iterations": 150,
    #     "convergence_threshold": 1e-8,
    # },
    # {
    #     "hidden_size": 30,
    #     "initial_temp": 500,
    #     "cooling_rate": 0.995,
    #     "max_iterations": 2000,
    #     "min_temp": 1e-6,
    #     "convergence_iterations": 100,
    #     "convergence_threshold": 1e-8,
    # },
    {
        "hidden_size": 30,
        "initial_temp": 1500,
        "cooling_rate": 0.99,
        "max_iterations": 20000,
        "min_temp": 1e-9,
        "convergence_iterations": 800,
        "convergence_threshold": 1e-10,
        # "cool_down_type": "exponential",
    },
]
compare_hyperparameters(file_path, "simulated_annealing", sa_params)

# Genetic Algorithm
ga_params = [
    # {
    #     "hidden_size": 30,
    #     "population_size": 50,
    #     "max_generations": 50,
    #     "mutation_rate": 0.1,
    #     "convergence_generations": 20,
    #     "convergence_threshold": 1e-8,
    # },
    # {
    #     "hidden_size": 30,
    #     "population_size": 100,
    #     "max_generations": 150,
    #     "mutation_rate": 0.05,
    #     "convergence_generations": 30,
    #     "convergence_threshold": 1e-8,
    # },
    # {
    #     "hidden_size": 30,
    #     "population_size": 400,
    #     "max_generations": 1500,
    #     "mutation_rate": 0.25,
    #     "convergence_generations": 125,
    #     "convergence_threshold": 1e-6,
    # },
]
compare_hyperparameters(file_path, "genetic_algorithm", ga_params)

# Randomized Hill Climbing
rhc_params = [
    # {"hidden_size": 30, "step_size": 0.1, "iterations": 1000},
    # {"hidden_size": 30, "step_size": 0.05, "iterations": 2000},
    # {"hidden_size": 30, "step_size": 0.0125, "iterations": 12000},
]
# compare_hyperparameters(file_path, "randomized_hill_climbing", rhc_params)

print("All plots have been saved in the 'plots' directory.")
