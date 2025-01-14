from global_files import public_variables

import torch
import torch.nn as nn
import torch.optim as optim

class FullyConnectedDNN2(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super(FullyConnectedDNN2, self).__init__()
        # Define the network layers (e.g., fully connected layers with ReLU activations)
        layers = []
        previous_layer_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_layer_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            previous_layer_size = hidden_size
        layers.append(nn.Linear(previous_layer_size, 1))  # Output layer
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

    def train_with_validation(self, train_loader, val_loader, num_epochs, optimizer, criterion, device, patience=20):
        self.to(device)
        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_losses.append(running_loss / len(train_loader))

            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))

            # Early stopping logic
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model_state = self.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break  # Stop training early if no improvement for `patience` epochs

            # Optionally print progress
            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

         # Restore the best model state
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            
        return train_losses, val_losses

    def train_without_validation(self, train_loader, num_epochs, optimizer, criterion, device):
        self.to(device)
        train_losses = []

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_losses.append(running_loss / len(train_loader))

            # Optionally print progress
            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}")
        
        return train_losses


class WeightedMSELoss(nn.Module):
    def __init__(self, bin_weights):
        super(WeightedMSELoss, self).__init__()
        self.bin_weights = torch.tensor(bin_weights, dtype=torch.float32)

    def forward(self, predictions, targets, bins):
        sample_weights = self.bin_weights[bins]
        loss = sample_weights * (predictions - targets) ** 2
        return loss.mean()