# Created by jing at 26.04.24



import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from nesy_pi.aitk.utils import draw_utils, args_utils

# Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Plotting
def main():
    # Model, loss function, and optimizer
    args = args_utils.get_args()
    device = args.device
    num_epochs = 100
    data = torch.load(args.trained_model_folder / f"strategy_data.pth")
    train_X = torch.cat([d['X'][:2000] for d in data], dim=0)
    train_y = torch.cat([d['y'][:2000] for d in data], dim=0)

    perm_indices = torch.randperm(len(train_X))
    train_X = train_X[perm_indices]
    train_y = train_y[perm_indices]
    train_size = int(len(train_X) * 0.8)
    x_tensor, x_test_tensor = train_X[:train_size], train_X[train_size:]
    y_tensor, y_test_tensor = train_y[:train_size], train_y[train_size:]

    input_size = x_tensor.shape[1]
    output_size = y_tensor.shape[1]
    model = NeuralNetwork(input_size, output_size).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training the model
    bz = 1000
    loss = torch.tensor(0).to(device)
    loss_history = []
    test_loss_history = []
    for epoch in tqdm(range(num_epochs)):
        for i in range(int(train_size / bz)):
            # Forward pass
            l_i = i * bz
            r_i = (i + 1) * bz
            outputs = model(x_tensor[l_i:r_i].to(device)).to(device)
            loss = criterion(outputs, y_tensor[l_i:r_i].to(device)).to(device)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        # Evaluation
        with torch.no_grad():
            outputs = model(x_test_tensor.to(device)).to(device)
            test_loss = criterion(outputs, y_test_tensor.to(device)).to(device)
            test_loss_history.append(test_loss.item())
    print(f'Final loss: {loss.item():.4f}')
    print(f'Final test loss: {test_loss.item():.4f}')
    torch.save(model.state_dict(), args.trained_model_folder / 'strategy.pth')
    # Plot the loss history
    draw_utils.plot_line_chart(torch.tensor(loss_history).unsqueeze(0),
                               args.trained_model_folder, ['loss'], title=f'strategy_loss')
    draw_utils.plot_line_chart(torch.tensor(test_loss_history).unsqueeze(0),
                               args.trained_model_folder, ['test_loss'], title=f'strategy_test_loss')


    return model
if __name__ == "__main__":
    main()
