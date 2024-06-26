import torch
import torch.nn as nn


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


class AreaNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(AreaNet, self).__init__()
        hidden_dim = 32

        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, rho, phi):

        x = torch.cat((rho, phi))

        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)
        x = torch.sigmoid(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=256):
        super(MLP, self).__init__()
        # Number of input features is input_dim.
        self.layer_1 = nn.Linear(in_channels, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)
        x = torch.sigmoid(x)
        return x
