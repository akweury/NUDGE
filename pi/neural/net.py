# Created by shaji at 06/12/2023

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi.neural.layers import Conv


class WeightPredModel(nn.Module):
    def __init__(self, in_channels, label_channels, kernel=(3, 3), stride=(1, 1), padding=(0, 0)):
        super(WeightPredModel, self).__init__()

        self.d_conv1 = Conv(in_channels, 2 * in_channels, kernel, stride, padding)
        self.d_conv2 = Conv(2 * in_channels, 4 * in_channels, kernel, stride, padding)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(2 * 2 * 4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, label_channels)

    def forward(self, x):
        # input logic state: batch_size * 4 * 6
        x = self.d_conv1(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def choose_net(net_name, args):
    if net_name == "simple_nn":
        return WeightPredModel(in_channels=1, label_channels=len(args.action_name_getout))
    else:
        raise ValueError
