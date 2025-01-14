from global_files import public_variables

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from DNN import FullyConnectedDNN_class
torch.manual_seed(42)
# Load the data
df = pd.read_csv(public_variables.dfs_descriptors_only_path_ / '0ns.csv')

# Step 2: Split into features and target
X = df.drop(columns=['PKI', 'mol_id', 'conformations (ns)'], axis=1, errors='ignore')
y = df['PKI'].values.reshape(-1, 1)

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Bin target values to handle imbalance
bins = 5
binned_y, bin_edges = pd.cut(y.flatten(), bins=bins, labels=False, retbins=True)
dir_path = public_variables.base_path_ / 'code' / 'DNN' / 'train_val_loss' / 'test'
dir_path.mkdir(parents=True, exist_ok=True)
bin_counts = np.bincount(binned_y)  # Count samples per bin
total_samples = len(binned_y)
bin_weights = total_samples / (len(bin_counts) * bin_counts)  # Normalize weights

exp_weights = bin_weights**0.5  # Adjust exponent for desired smoothing
exp_weights /= exp_weights.max()  # Normalize to [0, 1]
test_size = 0.2
# Step 5: Split into training and testing sets
# Perform an 80%-20% train-test split
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X_scaled, y, np.arange(len(y)), test_size=test_size, random_state=42, stratify=binned_y
)
# Step 1: Plot the distribution of pKi for train and test sets
plt.figure(figsize=(10, 6))

# Plot for training set
plt.hist(y_train, bins=30, alpha=0.5, label='Training Set', color='blue', edgecolor='black')

# Plot for test set
plt.hist(y_test, bins=30, alpha=0.5, label='Test Set', color='red', edgecolor='black')

# Add labels and title
plt.xlabel('pKi')
plt.ylabel('Frequency')
plt.title('Distribution of pKi (y) in Training and Test Sets')
plt.legend()

# Show the plot
plt.savefig(dir_path / 'image.png' )
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_bins_tensor = torch.tensor(binned_y[train_idx], dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
test_bins_tensor = torch.tensor(binned_y[test_idx], dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_bins_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_bins_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Output summary
print(f"Training set size: {len(train_dataset)}")
print(f"Testing set size: {len(test_dataset)}")

hidden_layers = [64,64,64]
dropout_rate = 0.1
learning_rate = 1e-3
use_weights = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_epochs = 1000


model = FullyConnectedDNN_class.FullyConnectedDNN(
    input_size=X_train.shape[1],
    hidden_layers=hidden_layers,
    dropout_rate=dropout_rate
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set criterion based on whether weights are used
if use_weights:
    criterion = FullyConnectedDNN_class.WeightedMSELoss(bin_weights=torch.tensor(exp_weights, dtype=torch.float32).to(device))
else:
    criterion = nn.MSELoss()

# Initialize lists to track losses
all_train_losses = []
all_val_losses = []

# Training
# Training
all_train_losses = []  # Track training loss for analysis (optional)
for epoch in range(train_epochs):
    model.train()
    train_losses = []
    for X_batch, y_batch, bin_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        
        # Calculate loss
        if use_weights:
            bin_batch = bin_batch.to(device)
            loss = criterion(predictions, y_batch, bin_batch)
        else:
            loss = criterion(predictions, y_batch)
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    # Append average train loss for the epoch
    avg_train_loss = np.mean(train_losses)
    all_train_losses.append(avg_train_loss)

    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0 or epoch == train_epochs - 1:
        print(f"Epoch {epoch+1}/{train_epochs}, Train Loss: {avg_train_loss:.4f}")


model.eval()
test_predictions = []
test_targets = []

with torch.no_grad():
    for X_batch, y_batch, _ in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = model(X_batch)
        test_predictions.extend(predictions.cpu().numpy())
        test_targets.extend(y_batch.cpu().numpy())

# Convert to numpy arrays
test_predictions = np.array(test_predictions).flatten()
test_targets = np.array(test_targets).flatten()

# Calculate metrics
mse = mean_squared_error(test_targets, test_predictions)
r2 = r2_score(test_targets, test_predictions)
# Step 1: Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(test_targets, test_predictions, alpha=0.6, color='blue', edgecolor='black')

# Step 2: Add a line of perfect prediction
plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], color='red', linestyle='--', label="Perfect Prediction")

# Step 3: Add labels and title
plt.xlabel('True pKi')
plt.ylabel('Predicted pKi')
plt.title('Predicted vs True pKi (Test Set)')

# Step 4: Show legend
plt.legend()

# Step 5: Show the plot
plt.savefig(dir_path / f'imagepredtrue_{test_size}_{train_epochs}_{use_weights}.png')
print(f"Test MSE: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")