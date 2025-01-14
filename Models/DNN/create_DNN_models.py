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

bin_counts = np.bincount(binned_y)  # Count samples per bin
total_samples = len(binned_y)
bin_weights = total_samples / (len(bin_counts) * bin_counts)  # Normalize weights

exp_weights = bin_weights**0.5  # Adjust exponent for desired smoothing
exp_weights /= exp_weights.max()  # Normalize to [0, 1]

# Step 5: Split into training, validation, and testing sets
# Split into temp and test sets
X_temp, X_test, y_temp, y_test, temp_idx, test_idx = train_test_split(
    X_scaled, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=binned_y
)

# Split temp set into training and validation sets
X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
    X_temp, y_temp, temp_idx, test_size=0.15, random_state=42, stratify=binned_y[temp_idx]
)  # Validation = 0.3 of 90% = 27%, Training = 63%

# Step 6: Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_bins_tensor = torch.tensor(binned_y[train_idx], dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
val_bins_tensor = torch.tensor(binned_y[val_idx], dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Step 7: Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_bins_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor, val_bins_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Output summary
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Testing set size: {len(test_dataset)}")

# Hyperparameter grid search
hidden_layer_options = [[128, 64], [256, 128], [128,64,32], [256,128,64], [64,32]]
dropout_rate_options = [0.1, 0.2, 0.25]
learning_rate_options = [1e-3, 1e-4, 1e-5]
hidden_layer_options = [[256, 128], [256,128,64]]
dropout_rate_options = [0]
learning_rate_options = [1e-3]
use_weights_options = [True]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_val_loss = float('inf')
best_model = None
best_hyperparams = {}

dir_path = public_variables.base_path_ / 'code' / 'DNN' / 'train_val_loss'
dir_path.mkdir(parents=True, exist_ok=True)

train_epochs = 100

for hidden_layers in hidden_layer_options:
    for dropout_rate in dropout_rate_options:
        for lr in learning_rate_options:
            for use_weights in use_weights_options:  # Loop over weight usage
                # Initialize model
                model = FullyConnectedDNN_class.FullyConnectedDNN(
                    input_size=X_train.shape[1],
                    hidden_layers=hidden_layers,
                    dropout_rate=dropout_rate
                ).to(device)
                
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # Set criterion based on whether weights are used
                if use_weights:
                    criterion = FullyConnectedDNN_class.WeightedMSELoss(bin_weights=torch.tensor(exp_weights, dtype=torch.float32).to(device))
                else:
                    criterion = nn.MSELoss()

                # Initialize lists to track losses
                all_train_losses = []
                all_val_losses = []

                # Training
                for epoch in range(train_epochs):
                    model.train()
                    train_losses = []
                    for X_batch, y_batch, bin_batch in train_loader:
                        X_batch, y_batch, bin_batch = X_batch.to(device), y_batch.to(device), bin_batch.to(device)
                        
                        optimizer.zero_grad()
                        predictions = model(X_batch)
                        
                        # Calculate loss
                        if use_weights:
                            loss = criterion(predictions, y_batch, bin_batch)
                        else:
                            loss = criterion(predictions, y_batch)
                        
                        loss.backward()
                        optimizer.step()
                        train_losses.append(loss.item())
                    
                    # Append average train loss for the epoch
                    avg_train_loss = np.mean(train_losses)
                    all_train_losses.append(avg_train_loss)
                    
                    # Validation
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for X_val_batch, y_val_batch, val_bin_batch in val_loader:
                            X_val_batch, y_val_batch, val_bin_batch = (
                                X_val_batch.to(device),
                                y_val_batch.to(device),
                                val_bin_batch.to(device),
                            )
                            val_preds = model(X_val_batch)
                            
                            # Calculate validation loss
                            if use_weights:
                                val_loss = criterion(val_preds, y_val_batch, val_bin_batch)
                            else:
                                val_loss = criterion(val_preds, y_val_batch)
                                
                            val_losses.append(val_loss.item())
                    
                    # Append average validation loss for the epoch
                    avg_val_loss = np.mean(val_losses)
                    all_val_losses.append(avg_val_loss)
                
                # Plot and save the loss graph for the current combination
                plt.figure()
                plt.plot(all_train_losses, label="Training Loss")
                plt.plot(all_val_losses, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Loss Graph - Hidden Layers: {hidden_layers}, Dropout Rate: {dropout_rate}, LR: {lr}, Use Weights: {use_weights}")
                plt.legend()
                # Save the figure with hyperparameter details
                filename = f"loss_graph_hl{hidden_layers}_dr{dropout_rate}_lr{lr}_uw{use_weights}.png"
                plt.savefig(public_variables.base_path_ / 'code' / 'DNN' / 'train_val_loss' / filename)
                plt.close()

                # Print the hyperparameters and validation loss after each evaluation
                avg_val_loss = np.mean(all_val_losses)
                print(f"Hidden Layers: {hidden_layers}, Dropout Rate: {dropout_rate}, "
                      f"Learning Rate: {lr}, Use Weights: {use_weights}, "
                      f"Validation Loss: {avg_val_loss}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model = model
                    best_hyperparams = {
                        'hidden_layers': hidden_layers,
                        'dropout_rate': dropout_rate,
                        'lr': lr,
                        'use_weights': use_weights  # Store the best setting for weights
                    }

# for hidden_layers in hidden_layer_options:
#     for dropout_rate in dropout_rate_options:
#         for lr in learning_rate_options:
#             for use_weights in use_weights_options:  # Loop over weight usage
#                 # Initialize model
#                 model = FullyConnectedDNN_class.FullyConnectedDNN(
#                     input_size=X_train.shape[1],
#                     hidden_layers=hidden_layers,
#                     dropout_rate=dropout_rate
#                 ).to(device)
                
#                 optimizer = optim.Adam(model.parameters(), lr=lr)
                
#                 # Set criterion based on whether weights are used
#                 if use_weights:
#                     criterion = FullyConnectedDNN_class.WeightedMSELoss(bin_weights=torch.tensor(exp_weights, dtype=torch.float32).to(device))
#                 else:
#                     criterion = nn.MSELoss()

#                 # Training
#                 for epoch in range(300):
#                     model.train()
#                     train_losses = []
#                     for X_batch, y_batch, bin_batch in train_loader:
#                         X_batch, y_batch, bin_batch = X_batch.to(device), y_batch.to(device), bin_batch.to(device)
                        
#                         optimizer.zero_grad()
#                         predictions = model(X_batch)
                        
#                         # Calculate loss
#                         if use_weights:
#                             loss = criterion(predictions, y_batch, bin_batch)
#                         else:
#                             loss = criterion(predictions, y_batch)
                        
#                         loss.backward()
#                         optimizer.step()
#                         train_losses.append(loss.item())
                
#                 # Validation
#                 model.eval()
#                 val_losses = []
#                 with torch.no_grad():
#                     for X_val_batch, y_val_batch, val_bin_batch in val_loader:
#                         X_val_batch, y_val_batch, val_bin_batch = (
#                             X_val_batch.to(device),
#                             y_val_batch.to(device),
#                             val_bin_batch.to(device),
#                         )
#                         val_preds = model(X_val_batch)
                        
#                         # Calculate validation loss
#                         if use_weights:
#                             val_loss = criterion(val_preds, y_val_batch, val_bin_batch)
#                         else:
#                             val_loss = criterion(val_preds, y_val_batch)
                            
#                         val_losses.append(val_loss.item())
                
#                 # Print the hyperparameters and validation loss after each evaluation
                
                
#                 # Track validation loss
#                 avg_val_loss = np.mean(val_losses)


#                 print(f"Hidden Layers: {hidden_layers}, Dropout Rate: {dropout_rate}, "
#                       f"Learning Rate: {lr}, Use Weights: {use_weights}, "
#                       f"Validation Loss: {avg_val_loss}")
                
#                 if avg_val_loss < best_val_loss:
#                     best_val_loss = avg_val_loss
#                     best_model = model
#                     best_hyperparams = {
#                         'hidden_layers': hidden_layers,
#                         'dropout_rate': dropout_rate,
#                         'lr': lr,
#                         'use_weights': use_weights  # Store the best setting for weights
#                     }

print("Best Hyperparameters:", best_hyperparams)
# Calculate R² for the best model on the validation set
best_model.eval()
with torch.no_grad():
    val_preds = best_model(torch.tensor(X_val, dtype=torch.float32).to(device))
    best_val_r2 = r2_score(y_val, val_preds.cpu().numpy())
print(f"Best Validation R²: {best_val_r2:.4f}")

# Discretize the target into bins
n_bins = 5  # Choose the number of bins you want
binned_y, bin_edges = pd.cut(y.flatten(), bins=bins, labels=False, retbins=True)

# Use StratifiedKFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
final_model_epochs = 1000
r2_scores_with_weights = []
r2_scores_without_weights = []
X = df.drop(columns=['PKI', 'mol_id', 'conformations (ns)'], axis=1, errors='ignore').values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Create TensorDataset for the dataset
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Step 4: Create TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)

# Initialize variables
outer_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
r2_scores = []
mse_scores = []

# Loop over folds
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled, binned_y)):  # X is the feature matrix, y_binned is the binned target
    # Split into train and test data using indices
    train_data = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    test_data = TensorDataset(X_tensor[test_idx], y_tensor[test_idx])
    
    print(f"Fold {fold+1}/{kf.get_n_splits()}")
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    # Initialize your model with the best hyperparameters
    model2 = FullyConnectedDNN_class.FullyConnectedDNN(
        input_size=X.shape[1],
        hidden_layers=best_hyperparams['hidden_layers'],
        dropout_rate=best_hyperparams['dropout_rate']
    ).to(device)
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(model2.parameters(), lr=best_hyperparams['lr'])
    criterion = nn.MSELoss()  # You can use WeightedMSELoss if you need

    # Train model
    model2.train_model(train_data, test_data, num_epochs=1000, optimizer=optimizer, criterion=criterion, device=device)

    # Evaluate on the test set and calculate R²
    model2.eval()  # Set the model to evaluation mode
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, targets in test_data:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model2(inputs)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    
    # Calculate R² score
    r2 = r2_score(y_true, y_pred)
    r2_scores.append(r2)
    print(f"Fold {fold+1} R² score: {r2:.4f}")

# Average R² across folds
mean_r2 = sum(r2_scores) / len(r2_scores)
print(f"Average R² across all folds: {mean_r2:.4f}")

# # Perform cross-validation
# for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
#     # Split into train and test data
#     train_data = DataLoader(dataset=train_idx, batch_size=32, shuffle=True)
#     test_data = DataLoader(dataset=test_idx, batch_size=32, shuffle=False)
#     print(train_data)
#     # Train model with weights
#     model_with_weights = FullyConnectedDNN_class.FullyConnectedDNN(
#         input_size=X.shape[1],
#         hidden_layers=best_hyperparams['hidden_layers'],
#         dropout_rate=best_hyperparams['dropout_rate']
#     ).to(device)
#     optimizer_with_weights = optim.Adam(model_with_weights.parameters(), lr=best_hyperparams['lr'])
#     criterion_with_weights = FullyConnectedDNN_class.WeightedMSELoss(bin_weights=torch.tensor(exp_weights, dtype=torch.float32).to(device))

#     # Train model
#     model_with_weights.train_model(train_data, test_data, num_epochs=300, optimizer=optimizer_with_weights, criterion=criterion_with_weights, device=device)
    
#     # Evaluate on the test set
#     model_with_weights.eval()
#     test_preds, test_targets = [], []
#     with torch.no_grad():
#         for inputs, targets in test_data:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model_with_weights(inputs)
#             test_preds.append(outputs.cpu().numpy())
#             test_targets.append(targets.cpu().numpy())
    
#     test_preds = np.concatenate(test_preds, axis=0)
#     test_targets = np.concatenate(test_targets, axis=0)

#     # Calculate R² score for this fold
#     r2_with_weights = r2_score(test_targets, test_preds)
#     r2_scores_with_weights.append(r2_with_weights)

#     # Train model without weights
#     model_without_weights = FullyConnectedDNN_class.FullyConnectedDNN(
#         input_size=train_data.shape[1],
#         hidden_layers=best_hyperparams['hidden_layers'],
#         dropout_rate=best_hyperparams['dropout_rate']
#     ).to(device)
#     optimizer_without_weights = optim.Adam(model_without_weights.parameters(), lr=best_hyperparams['lr'])
#     criterion_without_weights = nn.MSELoss()

#     # Train model without weights
#     model_without_weights.train_model(train_data, test_data, num_epochs=300, optimizer=optimizer_without_weights, criterion=criterion_without_weights, device=device)
    
#     # Evaluate on the test set
#     model_without_weights.eval()
#     test_preds, test_targets = [], []
#     with torch.no_grad():
#         for inputs, targets in test_data:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model_without_weights(inputs)
#             test_preds.append(outputs.cpu().numpy())
#             test_targets.append(targets.cpu().numpy())
    
#     test_preds = np.concatenate(test_preds, axis=0)
#     test_targets = np.concatenate(test_targets, axis=0)

#     # Calculate R² score for this fold
#     r2_without_weights = r2_score(test_targets, test_preds)
#     r2_scores_without_weights.append(r2_without_weights)

# # Print average R² scores for models with and without weights
# print(f"Average R² with weights: {r2_scores_with_weights}")
# print(f"Average R² without weights: {r2_scores_without_weights}")



















# kf = KFold(n_splits=10, shuffle=True, random_state=42)
# full_dataset = X
# # Initialize variables to track performance
# cv_results_with_weights = []
# cv_results_without_weights = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
#     train_data = full_dataset[train_idx]
#     val_data = full_dataset[val_idx]

#     # Create DataLoaders for this fold
#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

#     # Train model with weights
#     model_with_weights = FullyConnectedDNN_class.FullyConnectedDNN(
#         input_size=train_data.shape[1],
#         hidden_layers=best_hyperparams['hidden_layers'],
#         dropout_rate=best_hyperparams['dropout_rate']
#     ).to(device)
#     optimizer_with_weights = optim.Adam(model_with_weights.parameters(), lr=best_hyperparams['lr'])
#     criterion_with_weights = FullyConnectedDNN_class.WeightedMSELoss(bin_weights=torch.tensor(exp_weights, dtype=torch.float32).to(device))
#     train_losses, val_losses = model_with_weights.train_model(
#         train_loader, val_loader, num_epochs=300, optimizer=optimizer_with_weights, criterion=criterion_with_weights, device=device
#     )
#     cv_results_with_weights.append(evaluate_model(model_with_weights, val_loader))

#     # Train model without weights
#     model_without_weights = FullyConnectedDNN_class.FullyConnectedDNN(
#         input_size=train_data.shape[1],
#         hidden_layers=best_hyperparams['hidden_layers'],
#         dropout_rate=best_hyperparams['dropout_rate']
#     ).to(device)
#     optimizer_without_weights = optim.Adam(model_without_weights.parameters(), lr=best_hyperparams['lr'])
#     criterion_without_weights = nn.MSELoss()
#     train_losses, val_losses = model_without_weights.train_model(
#         train_loader, val_loader, num_epochs=300, optimizer=optimizer_without_weights, criterion=criterion_without_weights, device=device
#     )
#     cv_results_without_weights.append(evaluate_model(model_without_weights, val_loader))














# # Combine train and validation sets for final training
# final_train_loader = DataLoader(
#     TensorDataset(
#         torch.cat((X_train_tensor, X_val_tensor)), 
#         torch.cat((y_train_tensor, y_val_tensor)),
#         torch.cat((train_bins_tensor, val_bins_tensor))
#     ),
#     batch_size=32, 
#     shuffle=True
# )

# # Initialize the final model with best hyperparameters
# final_model = FullyConnectedDNN_class.FullyConnectedDNN(
#     input_size=X_train.shape[1],
#     hidden_layers=best_hyperparams['hidden_layers'],
#     dropout_rate=best_hyperparams['dropout_rate']
# ).to(device)

# criterion = FullyConnectedDNN_class.WeightedMSELoss(bin_weights=torch.tensor(exp_weights, dtype=torch.float32).to(device)) \
#     if best_hyperparams['use_weights'] else nn.MSELoss()
# optimizer = optim.Adam(final_model.parameters(), lr=best_hyperparams['lr'])

# # Train the final model
# epochs = 300
# training_losses = []
# validation_losses = []

# for epoch in range(epochs):
#     # Training
#     final_model.train()
#     train_losses = []
#     for X_batch, y_batch, bin_batch in final_train_loader:
#         X_batch, y_batch, bin_batch = X_batch.to(device), y_batch.to(device), bin_batch.to(device)

#         optimizer.zero_grad()
#         predictions = final_model(X_batch)
#         loss = criterion(predictions, y_batch, bin_batch)
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())
#     training_losses.append(np.mean(train_losses))

# # Predictions vs True Values on Test Set
# final_model.eval()
# test_preds, test_true = [], []
# with torch.no_grad():
#     for X_test_batch, y_test_batch in test_loader:
#         X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
#         predictions = final_model(X_test_batch)
#         test_preds.extend(predictions.cpu().numpy())
#         test_true.extend(y_test_batch.cpu().numpy())

# # Plot Predictions vs True Values
# plt.figure(figsize=(10, 6))
# plt.scatter(test_true, test_preds, alpha=0.6, color='green')
# plt.plot([min(test_true), max(test_true)], [min(test_true), max(test_true)], 'r--', label='Perfect Fit')
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Predicted vs True Values')
# plt.legend()
# plt.savefig('predicted_vs_true_test.png')
# plt.close()











# model = FullyConnectedDNN_class.FullyConnectedDNN(input_size, hidden_layers, output_size, dropout_rate)

# # Step 9: Define optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()

# # Step 10: Train the model
# epochs = 3000
# model.train()

# for epoch in range(epochs):
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         predictions = model(batch_X).squeeze()
#         loss = criterion(predictions, batch_y.squeeze())
#         loss.backward()
#         optimizer.step()
    
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# # Step 11: Evaluate the model
# model.eval()
# with torch.no_grad():
#     predictions = model(X_test_tensor).squeeze()
#     predicted_values = predictions.numpy()
#     true_values = y_test_tensor.numpy()

# # Step 12: Calculate R^2 score
# r2 = r2_score(true_values, predicted_values)
# print(f"R^2 Score: {r2:.4f}")

# # Plot predicted vs. true values
# plt.figure(figsize=(8, 6))
# plt.scatter(true_values, predicted_values, alpha=0.6, color='blue', label='Predicted vs True')
# plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 
#          color='red', linestyle='--', label='Ideal Fit (y = x)')
# plt.title('Predicted vs. True Values')
# plt.xlabel('True Values (y)')
# plt.ylabel('Predicted Values (y)')
# plt.legend()
# plt.grid(True)
# plt.savefig(public_variables.base_path_ / 'code' / 'DNN' / '1ns_pred_vs_real_DNN2.png')




# Hyperparameters to experiment with
# learning_rates = [0.0001, 0.001, 0.01]
# batch_sizes = [32, 64]
# dropout_rates = [0.2, 0.3, 0.4]
# hidden_layers_options = [
#     [128, 64],
#     [256, 128],
#     [128, 64, 32]
# ]

# # Hyperparameter tuning loop
# for lr in learning_rates:
#     for batch_size in batch_sizes:
#         for dropout_rate in dropout_rates:
#             for hidden_layers in hidden_layers_options:
#                 # Create DataLoader with current batch size
#                 train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#                 # Instantiate model
#                 model = FullyConnectedDNN_class.FullyConnectedDNN(input_size=X_train.shape[1], hidden_layers=hidden_layers, output_size=1, dropout_rate=dropout_rate)
#                 optimizer = optim.Adam(model.parameters(), lr=lr)
#                 criterion = nn.MSELoss()

#                 # Training loop
#                 num_epochs = 30
#                 for epoch in range(num_epochs):
#                     model.train()
#                     running_loss = 0.0
#                     for inputs, targets in train_loader:
#                         optimizer.zero_grad()
#                         outputs = model(inputs)
#                         loss = criterion(outputs, targets)
#                         loss.backward()
#                         optimizer.step()
#                         running_loss += loss.item()

#                     # Print loss every 5 epochs
#                     if (epoch + 1) % 5 == 0:
#                         print(f"LR: {lr}, Batch size: {batch_size}, Dropout: {dropout_rate}, Hidden layers: {hidden_layers}, Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
                
#                 # Evaluate the model on the test set
#                 model.eval()
#                 with torch.no_grad():
#                     y_pred = model(X_test_tensor)
#                     test_loss = criterion(y_pred, y_test_tensor)
#                     print(f"Test Loss for LR: {lr}, Batch size: {batch_size}, Dropout: {dropout_rate}, Hidden layers: {hidden_layers}: {test_loss.item()}\n")
