# Created by shaji at 06/12/2023

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from sklearn.neighbors import KernelDensity

# Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


def generate_data(pos_data, gen_num):
    """ generate more data following a multivariate normal distribution with the calculated mean and covariance. """

    # Perform kernel density estimation
    kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
    kde.fit(pos_data)
    # Generate new points
    new_points = torch.from_numpy(kde.sample(gen_num)).to(torch.float32)
    return new_points


# Plotting
def fit_classifier(x_tensor, y_tensor, num_epochs, device):
    # Model, loss function, and optimizer
    input_size = 3
    model = NeuralNetwork(input_size).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training the model

    loss = torch.tensor(0).to(device)

    for epoch in tqdm(range(num_epochs), desc=f'Fit Classifier'):
        # Forward pass
        outputs = model(x_tensor.to(device)).to(device)
        loss = criterion(outputs, y_tensor.to(device)).to(device)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Final loss: {loss.item():.4f}')
    return model
