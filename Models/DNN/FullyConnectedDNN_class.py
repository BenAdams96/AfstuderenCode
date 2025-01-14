from global_files import public_variables

import torch
import torch.nn as nn
import torch.optim as optim

class FullyConnectedDNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=1, dropout_rate=0.2):
        super(FullyConnectedDNN, self).__init__()
        layers = []
        in_features = input_size
        
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())  # Hardcoded ReLU
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_units
        
        layers.append(nn.Linear(in_features, output_size))  # Output layer
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

    def train_model(self, train_loader, val_loader, num_epochs, optimizer, criterion, device, patience=20):
        self.to(device)
        print(f"Training on device: {device}")
        
        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(num_epochs):
            # Training phase
            self.train()  # Set model to training mode once at the start of each epoch
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
            self.eval()  # Set model to evaluation mode once at the start of the validation phase
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient tracking for validation
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))

            # # Check if the validation loss improved
            # if val_losses[-1] < best_val_loss:
            #     best_val_loss = val_losses[-1]
            #     epochs_without_improvement = 0
            #     best_model_state = self.state_dict()  # Save the current model state
            # else:
            #     epochs_without_improvement += 1

            # # Early stopping if no improvement for 'patience' epochs
            # if epochs_without_improvement >= patience:
            #     print(f"Early stopping triggered at epoch {epoch+1}")
            #     break
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        # # Load the best model state
        # if best_model_state is not None:
        #     self.load_state_dict(best_model_state)

            
        
        return train_losses, val_losses
    
    def train_model2(self, train_loader, val_loader, num_epochs, optimizer, criterion, device, patience=20):
        self.to(device)
        print(f"Training on device: {device}")
        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        return train_losses, val_losses

class WeightedMSELoss(nn.Module):
    def __init__(self, bin_weights):
        super(WeightedMSELoss, self).__init__()
        self.bin_weights = torch.tensor(bin_weights, dtype=torch.float32)

    def forward(self, predictions, targets, bins):
        sample_weights = self.bin_weights[bins]
        loss = sample_weights * (predictions - targets) ** 2
        return loss.mean()