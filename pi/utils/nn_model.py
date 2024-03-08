# Created by shaji at 06/12/2023

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from sklearn.neighbors import KernelDensity
from pi.utils import draw_utils


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
    kde = KernelDensity(bandwidth=0.001, kernel='gaussian')
    kde.fit(pos_data)
    # Generate new points
    new_points = torch.from_numpy(kde.sample(gen_num)).to(torch.float32)
    # print(f"max, new: {new_points[:, 2].max()}, old:{pos_data[:,2].max()}")
    # print(f"min new : {new_points[:, 2].min()}, old: {pos_data[:,2].min()}")
    #

    return new_points


# Plotting
def fit_classifier(x_tensor, y_tensor, num_epochs, device, classifier_type, plot_path):
    # Model, loss function, and optimizer
    input_size = 3
    model = NeuralNetwork(input_size).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training the model

    loss = torch.tensor(0).to(device)
    loss_history = []
    for epoch in tqdm(range(num_epochs), desc=f'{classifier_type}'):
        # Forward pass
        outputs = model(x_tensor.to(device)).to(device)
        loss = criterion(outputs, y_tensor.to(device)).to(device)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    print(f'Final loss: {loss.item():.4f}')
    # Plot the loss history
    draw_utils.plot_line_chart(torch.tensor(loss_history).unsqueeze(0),
                               plot_path, ['loss'], title=f'loss_{classifier_type}')
    return model


# Define a simple neural network for state prediction
class StatePredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(StatePredictor, self).__init__()
        self.fc = nn.Linear(input_size + 1, output_size)  # +1 for action

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.fc(x)
        return x


def train_state_predictor(current_states, actions, next_states, args):
    # Model, loss function, and optimizer
    input_size = current_states.shape[1] * current_states.shape[2]
    output_size = current_states.shape[1] * current_states.shape[2]
    model = StatePredictor(input_size, output_size).to(args.device)
    criterion = nn.MSELoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training the model

    loss = torch.tensor(0).to(args.device)
    loss_history = []
    for epoch in tqdm(range(args.train_nn_epochs)):
        # Randomly sample a batch from the training data
        indices = np.random.choice(len(current_states), args.batch_size, replace=False)
        batch_current_states = current_states[indices].reshape(args.batch_size, -1)
        batch_actions = actions[indices].to(torch.float32).unsqueeze(1)
        batch_next_states_target = next_states[indices]

        # Zero the gradients, forward, backward, and optimize
        optimizer.zero_grad()
        predicted_next_states = model(batch_current_states, batch_actions)
        loss = criterion(predicted_next_states, batch_next_states_target.view(args.batch_size, -1))
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    print(f'Final loss: {loss.item():.4f}')
    # Plot the loss history
    draw_utils.plot_line_chart(torch.tensor(loss_history[100:]).unsqueeze(0),
                               args.check_point_path, ['loss'], title=f'loss_state_predictor', log_y=True)
    return model
