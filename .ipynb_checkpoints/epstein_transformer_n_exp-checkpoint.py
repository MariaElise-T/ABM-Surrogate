import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#from torch.optim import Adam
#import torch.nn.functional as F
import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe):
        self.X = torch.tensor(dataframe.iloc[:, :3].values, dtype=torch.float32)  # Input: [Batch, 3]
        self.Y = torch.tensor(dataframe.iloc[:, 3:].values, dtype=torch.float32).unsqueeze(-1) # Output: [Batch, 255, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

# Transformer Model
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerTimeSeriesModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_length = seq_length
 
        # Input Encoder (maps input to d_model size)
        self.encoder = nn.Linear(input_dim, d_model)  # (Batch, 3) -> (Batch, d_model)
        
        # Project input to match the sequence length
        self.expand_input = nn.Linear(d_model, seq_length * d_model)  # (Batch, d_model) -> (Batch, seq_length * d_model)
        
        # Target embedding for decoder input
        self.target_embedding = nn.Linear(1, d_model)  # New embedding layer for target sequence
  
        # Positional Encoding for Time Steps
        self.pos_encoder = PositionalEncoding(d_model, seq_length)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final Output Layer
        self.output_layer = nn.Linear(d_model, output_dim)  # (Batch, 255, d_model) -> (Batch, 255, 1)

    def forward(self, x, target_seq):
        # x: Input features [Batch, 3]
        # target_seq: Target sequence for teacher forcing [Batch, 255, 1]
        
        # Encode input features
        encoded_input = self.encoder(x)  # [Batch, d_model]
        
        # Expand input to match sequence length
        expanded_input = self.expand_input(encoded_input)  # [Batch, seq_length * d_model]
        expanded_input = expanded_input.view(-1, self.seq_length, self.d_model)  # Reshape to [Batch, 255, d_model]
        
        # Add Positional Encoding
        expanded_input = self.pos_encoder(expanded_input)
        
        # Process the target sequence through the same encoding pipeline
  #      target_embeddings = self.encoder(target_seq)
  #      target_embeddings = nn.Linear(1, d_model)(target_seq)  # [Batch, 255, d_model]
        target_embeddings = self.target_embedding(target_seq)  # [Batch, 255, d_model]
        target_embeddings = self.pos_encoder(target_embeddings)
        
        # Decode sequence
        output = self.transformer_decoder(
            tgt=target_embeddings, memory=expanded_input
        )  # Output shape: [Batch, 255, d_model]
        
        # Map to output dimensions
        predictions = self.output_layer(output)  # [Batch, 255, 1]
        return predictions
    
def train_model(model, dataloader, optimizer, loss_fn, num_epochs, device):
    loss_list = list()
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            x, y = batch  # x: [Batch, N], y: [Batch, T]
            x, y = x.to(device), y.to(device)
            
            # Prepare target for teacher forcing
            target_seq = y 
            #target_seq = y[:, :-1]  # All except last time step
            #actual = y[:, 1:]       # All except first time step
            
            # Forward pass
            output = model(x, target_seq)
            loss = loss_fn(output, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    return loss_list

##################################################################
# Set up      
##################################################################

print("Importing data...\n")

data_output = pd.read_csv("C:/Users/met48/Desktop/TS-Clustering/SimData/epsteinCV_outputs_active.csv", header=None)
#print("Max value is :", data_output.to_numpy().max(), "\n")

scaler = MinMaxScaler()
#scaler.data_max_ = 1600
scaler.fit(data_output)
data_output = scaler.transform(data_output)
data_output = pd.DataFrame(data_output)

data_input = pd.read_csv("C:/Users/met48/Desktop/TS-Clustering/SimData/epsteinCV_inputs.csv", sep=" ", header=None)
data = pd.concat([data_input, data_output], axis=1)

data = data.sample(n=1000, random_state=1)

# Split the data into training and validation sets
train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the validation set to a new CSV file
valid_data.to_csv("validation_set_epstein_n1000.csv", index=False)

##################################################################
# Model 1      
##################################################################

print("Starting model 1\n")

dataset = TimeSeriesDataset(train_data)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

input_dim = 3      # Number of input features
output_dim = 1     # Predicting one value per time step
seq_length = 252   # Number of time steps in output
d_model = 128      # Embedding dimension for Transformer
nhead = 4          # Number of attention heads
num_layers = 2     # Number of Transformer layers
dim_feedforward = 512  # Feedforward network size

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerTimeSeriesModel(
    input_dim, output_dim, seq_length, d_model, nhead, num_layers, dim_feedforward
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()  # Regression loss

# Training loop
num_epochs = 20  # Adjust based on dataset size and performance
loss = train_model(model, dataloader, optimizer, loss_fn, num_epochs, device)

df = pd.DataFrame(loss, columns=["loss"])
df.to_csv('transformer_adam_lr001_epstein_loss_n1000.csv', index=False)
torch.save(model.state_dict(), "transformer_adam_lr001_epstein_n1000.pth")

##################################################################
# Set up      
##################################################################

print("Importing data...\n")

data_output = pd.read_csv("C:/Users/met48/Desktop/TS-Clustering/SimData/epsteinCV_outputs_active.csv", header=None)
#print("Max value is :", data_output.to_numpy().max(), "\n")

scaler = MinMaxScaler()
#scaler.data_max_ = 1600
scaler.fit(data_output)
data_output = scaler.transform(data_output)
data_output = pd.DataFrame(data_output)

data_input = pd.read_csv("C:/Users/met48/Desktop/TS-Clustering/SimData/epsteinCV_inputs.csv", sep=" ", header=None)
data = pd.concat([data_input, data_output], axis=1)

data = data.sample(n=5000, random_state=1)

# Split the data into training and validation sets
train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the validation set to a new CSV file
valid_data.to_csv("validation_set_epstein_n5000.csv", index=False)

##################################################################
# Model 2  
##################################################################

print("Starting model 1\n")

dataset = TimeSeriesDataset(train_data)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

input_dim = 3      # Number of input features
output_dim = 1     # Predicting one value per time step
seq_length = 252   # Number of time steps in output
d_model = 128      # Embedding dimension for Transformer
nhead = 4          # Number of attention heads
num_layers = 2     # Number of Transformer layers
dim_feedforward = 512  # Feedforward network size

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerTimeSeriesModel(
    input_dim, output_dim, seq_length, d_model, nhead, num_layers, dim_feedforward
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()  # Regression loss

# Training loop
num_epochs = 20  # Adjust based on dataset size and performance
loss = train_model(model, dataloader, optimizer, loss_fn, num_epochs, device)

df = pd.DataFrame(loss, columns=["loss"])
df.to_csv('transformer_adam_lr001_epstein_loss_n5000.csv', index=False)
torch.save(model.state_dict(), "transformer_adam_lr001_epstein_n5000.pth")