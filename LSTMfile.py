# import numpy as np
# import pandas as pd
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #to get rid of tensorflow messages
# import tensorflow as tf
# from keras import Sequential
# from keras.layers import LSTM, Dense
# # from keras import Sequential
# # from tensorflow.python.keras.layers import Dense, LSTM
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# import public_variables
# from glob import glob


# def reduce_conformations(df, interval=1):
#     """
#     Reduces the number of conformations per molecule in the dataframe
#     by selecting only specific conformations at given intervals, excluding 0.
    
#     Parameters:
#         df (pd.DataFrame): The large dataframe containing all conformations.
#         interval (float): The desired interval for selection, default is 1ns.
    
#     Returns:
#         pd.DataFrame: A reduced dataframe with only the specified conformations per molecule.
#     """
#     # Define the target conformations, starting from the first interval, excluding 0
#     target_conformations = [round(i * interval, 2) for i in range(1, int(10 / interval) + 1)]
    
#     # Filter the dataframe to only include rows with conformations in target_conformations
#     reduced_df = df[df['conformations (ns)'].isin(target_conformations)].copy(False)
    
#     return reduced_df


# # Step 1: Load and stack data from CSV files into a 3D array


import torch
import torch.nn as nn
import torch.optim as optim

# Define the LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # LSTM layer
        out, _ = self.lstm(x)  # output and hidden states
        # Fully connected layer (to output size)
        out = self.fc(out[:, -1, :])  # take the last time step's output
        return out

# initial_df = pd.read_csv(public_variables.initial_dataframe)
# step_size = 1.0
# filtered_df = reduce_conformations(initial_df, interval=1)
# print(filtered_df.shape)

# public_variables.LSTM_master_.mkdir(parents=True, exist_ok=True)
# filtered_df2 = filtered_df.sort_values(by=['mol_id', 'conformations (ns)']).reset_index(drop=True)


# filtered_df.to_csv(public_variables.LSTM_master_ / f'initial_dataframe.csv', index=False)
# filtered_df2.to_csv(public_variables.LSTM_master_ / f'initial_dataframe2.csv', index=False)

# Hyperparameters
input_size = 1      # Number of features in the input
hidden_size = 50    # Number of features in the hidden state
output_size = 1     # Single output per sequence
num_layers = 1      # Number of LSTM layers
num_epochs = 20
learning_rate = 0.001

# Create the model
model = SimpleLSTM(input_size, hidden_size, output_size, num_layers)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Sample Data
# Here, we create a random tensor to simulate a time-series input
# Replace this with your actual data tensor of shape (batch_size, seq_length, input_size)
seq_length = 10  # Length of each input sequence
batch_size = 5   # Number of sequences per batch
X_sample = torch.randn(batch_size, seq_length, input_size)
y_sample = torch.randn(batch_size, output_size)

# Training loop
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_sample)
    loss = criterion(outputs, y_sample)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
