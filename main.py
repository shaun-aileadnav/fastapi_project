import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb

import os

import random

TRAIN_DATA_PATH = "data/training.pt"
TEST_DATA_PATH = "data/test.pt"

# IMPORT IMAGE DATA
x, y = torch.load(TRAIN_DATA_PATH)

# ONE HOT ENCODER

y_original = torch.tensor([2, 4, 3, 0, 1])
y_new = F.one_hot(y, num_classes=10)

# RESHAPE IMAGE DATA
x.view(-1, 28**2).shape

# PYTORCH DATASET OBJECT


class CTDataset(Dataset):

    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255.
        self.y = F.one_hot(self.y, num_classes=10).to(float)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


train_ds = CTDataset(os.environ.get('TRAIN_DATA_PATH', 'data/training.pt'))
test_ds = CTDataset(os.environ.get('TEST_DATA_PATH', 'data/test.pt'))

xs, ys = train_ds[0:4]

# DATALOADER OBJECT

train_dl = DataLoader(train_ds, batch_size=5)

# CROSS ENTROPY LOSS

L = nn.CrossEntropyLoss()

# THE NETWORK


class MyNeuralNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, 100)
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50, 10)
        self.R = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()


f = MyNeuralNet()

# TRAINING


def train_model(dl, f, n_epochs=20):
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()
    # Train model
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        epoch_loss = 0.0
        for i, (x, y) in enumerate(dl):
            opt.zero_grad()
            loss_value = L(f(x), y)
            loss_value.backward()
            opt.step()
            epoch_loss += loss_value.item()

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss / len(dl)
        })

    return f  # Return the trained model


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == torch.argmax(y, dim=1)).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    run = wandb.init(project="mnist_classification", config={
        "architecture": "MyNeuralNet",
        "dataset": "MNIST",
        "epochs": 20,
        "learning_rate": 0.01,
        "batch_size": 5
    })

    train_ds = CTDataset(os.environ.get('TRAIN_DATA_PATH', 'data/training.pt'))
    test_ds = CTDataset(os.environ.get('TEST_DATA_PATH', 'data/test.pt'))

    train_dl = DataLoader(train_ds, batch_size=5)
    test_dl = DataLoader(test_ds, batch_size=5)

    model = MyNeuralNet()
    trained_model = train_model(train_dl, model, n_epochs=20)

    # Evaluate the model
    train_accuracy = evaluate_model(trained_model, train_dl)
    test_accuracy = evaluate_model(trained_model, test_dl)

    # Log final metrics
    wandb.log({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    })

    # Create a model artifact
    model_artifact = wandb.Artifact("my_model", type="model")

    # Define the model file path
    model_file_path = "models/my_model.pth"

    # Save the model state dictionary
    torch.save(trained_model.state_dict(), model_file_path)

    # Add the model file to the artifact
    model_artifact.add_file(model_file_path)

    # Log the artifact
    wandb.log_artifact(model_artifact)

    # Link the artifact to the Model Registry
    run.link_artifact(model_artifact, "ai-leadnav-org/wandb-registry-model/<Shauns_models>")

    # Finish the run
    wandb.finish()


if __name__ == "__main__":
    main()
