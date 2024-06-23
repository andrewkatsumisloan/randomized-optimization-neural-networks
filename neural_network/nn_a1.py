import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    X = data.iloc[:, :-1].values  # Features
    Y = data.iloc[:, -1].values  # Target (last column)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, Y_train, Y_test


class ANN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ANN, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden_layer(x))
        x = self.sigmoid(self.output_layer(x))
        return x


def train_and_evaluate_model(
    X_train, Y_train, X_test, Y_test, lr, num_epochs, hidden_size
):
    model = ANN(X_train.shape[1], hidden_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()

        outputs = model(X_train)
        loss = criterion(outputs, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_outputs = (outputs > 0.5).float()
        train_accuracy = (train_outputs == Y_train).sum().item() / Y_train.shape[0]
        train_accuracies.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, Y_test)

            test_outputs = (test_outputs > 0.5).float()
            test_accuracy = (test_outputs == Y_test).sum().item() / Y_test.shape[0]
            test_accuracies.append(test_accuracy)

        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
            )

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_outputs = (test_outputs > 0.5).float()
        accuracy = (test_outputs == Y_test).sum().item() / Y_test.shape[0]
        Y_test_np = Y_test.numpy()
        test_outputs_np = test_outputs.numpy()
        class_report = classification_report(
            Y_test_np,
            test_outputs_np,
            target_names=["Class 0", "Class 1"],
            output_dict=True,
        )
        return accuracy, class_report, model, train_accuracies, test_accuracies


def plot_accuracies(train_accuracies, test_accuracies, num_epochs, filename):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy vs Epochs For Gradient Descent NN")
    plt.legend()

    plt.savefig(filename)
    plt.show()


def main():
    file_path = "./dataset/diabetes_balanced.csv"

    filename = os.path.splitext(os.path.basename(file_path))[0] + "_accuracy_plot.png"
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data(file_path)

    lr = 0.01
    num_epochs = 55
    hidden_size = 30

    accuracy, class_report, model, train_accuracies, test_accuracies = (
        train_and_evaluate_model(
            X_train, Y_train, X_test, Y_test, lr, num_epochs, hidden_size
        )
    )

    print(f"Model accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(
        classification_report(
            Y_test.numpy(),
            (model(X_test) > 0.5).float().numpy(),
            target_names=["Class 0", "Class 1"],
        )
    )

    plot_accuracies(train_accuracies, test_accuracies, num_epochs, filename)


if __name__ == "__main__":
    main()
