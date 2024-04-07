# Created by shaji at 06/04/2024

# data.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class WordSequenceDataset(Dataset):
    def __init__(self, sequences, max_window_size=5):
        self.sequences = sequences
        self.max_window_size = max_window_size
        self.tokenizer = LabelEncoder()
        self.tokenizer.fit([word for seq in sequences for word in seq])
        self.vocab_size = len(self.tokenizer.classes_)

    def __len__(self):
        return sum(len(seq) - self.max_window_size for seq in self.sequences if len(seq) > self.max_window_size)

    def __getitem__(self, idx):
        seq_index = 0
        while True:
            seq = self.sequences[seq_index]
            if len(seq) > self.max_window_size:
                break
            seq_index += 1
        start_index = np.random.randint(0, len(seq) - self.max_window_size)
        input_sequence = self.tokenizer.transform(seq[start_index:start_index + self.max_window_size])
        target_word = self.tokenizer.transform([seq[start_index + self.max_window_size]])[0]
        return torch.LongTensor(input_sequence), torch.LongTensor([target_word])


# model.py
import torch
import torch.nn as nn


# Define positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout),
            num_layers)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src.transpose(0, 1))
        output = self.fc(output[-1])
        return output


# training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_model(model, dataloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# inference.py
import torch


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()


def predict_next_word(model, tokenizer, input_sequence):
    input_sequence_encoded = torch.LongTensor([tokenizer.transform(input_sequence)])
    with torch.no_grad():
        predicted_logits = model(input_sequence_encoded)
        # predicted_word_index = torch.argmax(predicted_logits, dim=1)
        return predicted_logits


# Define hyperparameters
embedding_size = 128
hidden_size = 256
num_layers = 2
num_heads = 2
dropout = 0.1
batch_size = 4
learning_rate = 0.001
epochs = 400


def train_transformer(dataset, model_file):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    model = TransformerModel(dataset.vocab_size, embedding_size, hidden_size, num_layers, num_heads, dropout)
    model.vocab_size = dataset.vocab_size
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, epochs)
    # Save the model
    torch.save(model.state_dict(), model_file)


def eval_transformer(input_sequence, dataset, model_file):
    # Load the model for inference
    loaded_model = TransformerModel(dataset.vocab_size, embedding_size, hidden_size, num_layers, num_heads, dropout)
    load_model(loaded_model, model_file)
    # Example of using the trained model for prediction
    predicted_word = predict_next_word(loaded_model, dataset.tokenizer, input_sequence)
    return predicted_word


def eval_pos_transformer(input_sequence, model_file):
    # Parameters
    input_size = 2  # Dimensionality of each 2D position
    output_size = 2  # Dimensionality of the next 2D point
    num_layers = 6
    hidden_size = 64
    num_heads = 8
    dropout = 0.1
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001

    # Define model, optimizer, and loss function
    # Create model
    model = RNNModel(input_size, hidden_size, output_size).to(input_sequence[0].device)

    load_model(model, model_file)
    model.eval()
    with torch.no_grad():
        input_positions = input_sequence.permute(1, 0, 2)  # Swap batch and sequence dimensions

        predicted_positions = model(input_positions)

    print("Predicted next position:", predicted_positions)

    return predicted_positions


import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


# Define the transformer model
class TransformerPositionPrediction(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, num_heads, dropout):
        super(TransformerPositionPrediction, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)  # Take the last output from the sequence
        return x


# Define your RNN-based model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)  # Take the last output from the sequence
        return out


from torch.nn.utils.rnn import pad_sequence


# Convert data to PyTorch DataLoader
def collate_fn(batch):
    sequences = [torch.tensor(item) for item in batch]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=float(0))
    return padded_sequences


def train_position_transformer(args, train_loader, val_loader, model_file):
    # Convert data to PyTorch DataLoader
    # Parameters
    input_size = 2  # Dimensionality of each 2D position
    output_size = 2  # Dimensionality of the next 2D point
    num_layers = 6
    hidden_size = 64
    num_heads = 2
    dropout = 0.1
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001

    # Define model, optimizer, and loss function

    # Create model
    model = RNNModel(input_size, hidden_size, output_size).to(args.device)

    # Define model, optimizer, and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 1000
    loss_history = torch.zeros(2, num_epochs)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)  # Target position is the last position in the sequence

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)
        loss_history[0, epoch] = train_loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)  # Target position is the last position in the sequence
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)
        loss_history[1, epoch] = val_loss
        print(f"val_loss: {val_loss:.4f}")
    from pi.utils import draw_utils
    draw_utils.plot_line_chart(loss_history, path=args.trained_model_folder, labels=["train","val"],
                               title=f"position_transformer",
                               figure_size=(10, 5))
    # Save the model
    torch.save(model.state_dict(), model_file)
    return model


def position_dataset(game_positions):
    # Remove lists with length less than 2
    points_list = [points for points in game_positions if len(points) >= 2]
    # Cut longer lists into smaller lists with length 2
    X = []
    y = []
    for points in points_list:
        if len(points) == 2:
            X.append(points[0].unsqueeze(0))
            y.append(points[1].unsqueeze(0))
        else:
            for i in range(len(points) - 1):
                X.append(points[i].unsqueeze(0))
                y.append(points[i + 1].unsqueeze(0))
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    return X, y
