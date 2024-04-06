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
        predicted_word_index = torch.argmax(predicted_logits, dim=1)
        predicted_word = tokenizer.inverse_transform(predicted_word_index.numpy().flatten())
        return predicted_word[0]


# Define hyperparameters
embedding_size = 128
hidden_size = 256
num_layers = 2
num_heads = 2
dropout = 0.1
batch_size = 4
learning_rate = 0.001
epochs = 1000


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
    print("Predicted next word:", predicted_word)
    return predicted_word


import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


# Define the transformer model
class TransformerPositionPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerPositionPrediction, self).__init__()
        config = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(config, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 2)  # Output 2D position

    def forward(self, input_positions):
        outputs = self.transformer(input_positions)
        pooled_output = outputs[1]  # Use the pooled output
        predicted_position = self.fc(pooled_output)
        return predicted_position


from torch.nn.utils.rnn import pad_sequence


# Convert data to PyTorch DataLoader
def collate_fn(batch):
    sequences = [torch.tensor(item) for item in batch]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=float(0))
    return padded_sequences


def train_position_transformer(train_data, val_data, model_file):
    # Convert data to PyTorch DataLoader
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=4, collate_fn=collate_fn)

    # Define model, optimizer, and loss function
    model = TransformerPositionPrediction(input_dim=2, hidden_dim=64, num_layers=2, num_heads=1).to(train_data[0].device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_positions = batch.permute(1, 0, 2)  # Swap batch and sequence dimensions
            target_positions = input_positions[-1]  # Target position is the last position in the sequence
            predicted_positions = model(input_positions)
            loss = criterion(predicted_positions, target_positions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * input_positions.size(0)
        train_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_positions = batch.permute(1, 0, 2)  # Swap batch and sequence dimensions
                target_positions = input_positions[-1]  # Target position is the last position in the sequence
                predicted_positions = model(input_positions)
                loss = criterion(predicted_positions, target_positions)
                val_loss += loss.item() * input_positions.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"val_loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), model_file)
    return model


def position_dataset(game_positions):

    # Remove lists with length less than 2
    points_list = [points for points in game_positions if len(points) >= 2]

    # Cut longer lists into smaller lists with length 2
    new_points_list = []
    for points in points_list:
        if len(points) == 2:
            new_points_list.append(points)
        else:
            for i in range(len(points) - 1):
                new_points_list.append(torch.cat((points[i].unsqueeze(0), points[i + 1].unsqueeze(0)), dim=0))
    return new_points_list