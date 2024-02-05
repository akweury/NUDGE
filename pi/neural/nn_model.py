# Created by shaji at 06/12/2023

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm


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

    # Calculate mean and covariance matrix from the original set of points
    # In general, for a covariance matrix to be positive definite,
    # it must be symmetric and have all positive eigenvalues.
    # For a collection of more than two points, the covariance matrix can be computed,
    # and it may or may not be positive definite depending on the distribution and relationships between the variables.
    assert len(pos_data) > 2
    mean = pos_data.mean(dim=0)
    covariance_matrix = torch.tensor(np.cov(pos_data.numpy(), rowvar=False), dtype=torch.double)
    # Generate new points following the same distribution

    generated_points = torch.distributions.MultivariateNormal(mean.double(), covariance_matrix).sample((gen_num,))


    generated_points = generated_points.to(torch.float32)
    return generated_points


# Plotting
def fit_classifier(x_tensor, y_tensor, num_epochs):
    # Model, loss function, and optimizer
    input_size = 2
    model = NeuralNetwork(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training the model

    loss = torch.tensor(0)

    for epoch in tqdm(range(num_epochs), desc=f'Fit Classifier'):
        # Forward pass
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Final loss: {loss.item():.4f}')
    return model
