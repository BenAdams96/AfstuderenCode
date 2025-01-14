from global_files import public_variables
from global_files import csv_to_dictionary

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

from Models.DNN import FullyConnectedDNN_class2
torch.manual_seed(42)

def DNN_function(name, df, dfs_path):

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

    # Initialize variables
    outer_folds = 5
    outer_kf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    r2_scores = []
    mse_scores = []

    hyperparameter_grid = [
        {"learning_rate": 0.001, "hidden_layers": [64, 32, 16],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [64, 32],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [128, 64, 32],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [128, 64],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [64, 32, 16],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [64, 32],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [128, 64, 32],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [128, 64],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [256, 128, 64],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [256, 128],"dropout_rate": 0.1},
        # Add more combinations as needed
    ]

    # Store losses and predictions across folds
    all_predictions = []
    all_true_values = []
    all_train_losses = []
    all_val_losses = []
    all_params = []
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X_scaled, binned_y)):
        print(f"Outer Fold {fold + 1}")

        # Split into train and test sets
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner split: Further split training data into 70% training, 10% validation
        X_inner_train, X_val, y_inner_train, y_val = train_test_split(
            X_train, y_train, test_size=0.125, stratify=pd.cut(y_train.flatten(), bins=bins, labels=False),
            random_state=42
        )

        # Convert to tensors for inner split
        inner_train_dataset = TensorDataset(torch.tensor(X_inner_train, dtype=torch.float32),
                                            torch.tensor(y_inner_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(inner_train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Hyperparameter tuning
        best_params = None
        best_val_loss = float('inf')
        for params in hyperparameter_grid:
            model = FullyConnectedDNN_class2.FullyConnectedDNN2(input_size=X_train.shape[1], hidden_layers=params["hidden_layers"],dropout_rate=params["dropout_rate"])
            optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
            criterion = torch.nn.MSELoss()

            # Train with validation to tune hyperparameters
            train_losses, val_losses = model.train_with_validation(
                train_loader, val_loader, num_epochs=150, optimizer=optimizer,
                criterion=criterion, device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # Track the best hyperparameters
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_params = params
                best_train_losses = train_losses
                best_val_losses = val_losses

        print(f"Best params for fold {fold + 1}: {best_params}, Validation Loss: {best_val_loss:.4f}")

        # Train final model on 80% training data with best hyperparameters
        final_train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.float32))
        final_train_loader = DataLoader(final_train_dataset, batch_size=32, shuffle=True)

        final_model = FullyConnectedDNN_class2.FullyConnectedDNN2(input_size=X_train.shape[1], hidden_layers=best_params['hidden_layers'],dropout_rate=best_params["dropout_rate"])
        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
        final_criterion = torch.nn.MSELoss()

        final_model.train_without_validation(
            final_train_loader, num_epochs=500, optimizer=final_optimizer,
            criterion=final_criterion, device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Evaluate on test set
        final_model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                    torch.tensor(y_test, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        predictions = []
        true_values = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
                outputs = final_model(inputs)
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(targets.numpy())
        # Append fold results
        all_predictions.extend(predictions)
        all_true_values.extend(true_values)
        all_train_losses.append(best_train_losses)
        all_val_losses.append(best_val_losses)
        all_params.append(best_params)
        # Calculate metrics
        mse = mean_squared_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        mse_scores.append(mse)
        r2_scores.append(r2)

        print(f"Outer Fold {fold + 1} - Test MSE: {mse:.4f}, Test R²: {r2:.4f}")

    # Print overall metrics
    print(f"Average MSE: {np.mean(mse_scores):.4f}")
    print(f"Average R²: {np.mean(r2_scores):.4f}")

    # fold_results.append(r2_score)

    data = pd.Series(fold_results)

    fold_results = {
        "mol_id": name,
        "mean_test_score": pd.Series(r2_scores).mean(),
        "std_test_score": pd.Series(r2_scores).std(),
        "params": [1,2],
        **{f"split{split_idx}_test_score": r2_scores[split_idx] for split_idx in range(outer_folds)},  # Loop to add splits
    }

    create_train_loss_plots(name, all_params, all_train_losses, all_val_losses, dfs_path)
    create_pred_true_plots(name, all_params, all_true_values, all_predictions, dfs_path)

    return fold_results

def create_train_loss_plots(name, all_params, all_train_losses, all_val_losses, dfs_path):

    # Plot Training and Validation Loss Curves
    model_ = 'DNN'
    Modelresults_path = dfs_path / f'ModelResults_{model_}'
    path = Modelresults_path / 'train_loss_plots'
    path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
        plt.plot(train_loss, label=f"Fold {fold + 1} - Train Loss")
        plt.plot(val_loss, label=f"Fold {fold + 1} - Validation Loss", linestyle="--")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if name == '0ns':
        plt.title(f"minimized conformation - Training and Validation Loss Across Folds")
    else:
        plt.title(f"{name} - Training and Validation Loss Across Folds")
    plt.legend()
    plt.savefig(path / f'{name}_trainloss_all_folds.png')
    plt.close()

    for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
        print(all_params)
        print(fold)
        print(all_params[fold])
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label=f"Train Loss")
        plt.plot(val_loss, label=f"Validation Loss", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        if name == '0ns':
            plt.title(f"minimized conformation -Fold {fold + 1} - Training and Validation Loss - {all_params[fold]}")
        else:
            plt.title(f"{name} -Fold {fold + 1} - Training and Validation Loss - {all_params[fold]}")
        plt.legend()
        plt.savefig(path / f'{name}_trainloss_fold_{fold + 1}.png')
        plt.close()
    return

def create_pred_true_plots(name, params, all_true_values, all_predictions, dfs_path):
    model_ = 'DNN'
    Modelresults_path = dfs_path / f'ModelResults_{model_}'
    path = Modelresults_path / 'pred_true_plots'
    path.mkdir(parents=True, exist_ok=True)
    # Plot Predicted vs True pKi Values
    plt.figure(figsize=(8, 8))
    plt.scatter(all_true_values, all_predictions, alpha=0.6, edgecolor='k')
    plt.plot([min(all_true_values), max(all_true_values)],
             [min(all_true_values), max(all_true_values)], color="red", linestyle="--", label="Ideal Fit")
    plt.xlabel("True pKi")
    plt.ylabel("Predicted pKi")
    if name == '0ns':
        plt.title(f"minimized conformation - Predicted vs True pKi Across All Folds")
    else:
        plt.title(f"{name} - Predicted vs True pKi Across All Folds")
    plt.legend()
    plt.savefig(path / f'{name}_pred_true.png')
    plt.close()
    return

def main(dfs_path = public_variables.dfs_descriptors_only_path_):
    descriptors = public_variables.Descriptor_

    model_ = 'DNN'
    Modelresults_path = dfs_path / f'ModelResults_{model_}' #CHECK: if 'RF' is in public_variables or something else perhaps
    Modelresults_path.mkdir(parents=True, exist_ok=True)

    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic(dfs_path, exclude_files=['conformations_1000.csv','conformations_1000_molid.csv','conformations_500.csv','conformations_200.csv','conformations_100.csv','conformations_50.csv','initial_dataframe.csv','initial_dataframes_best.csv','MD_output.csv']) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys())) #RDKIT first
    dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic} #order
    print(dfs_in_dic.keys())
    first_four_keys = list(dfs_in_dic.keys())[0:11] + [list(dfs_in_dic.keys())[14]] + [list(dfs_in_dic.keys())[16]] + [list(dfs_in_dic.keys())[17]] #+ list(dfs_in_dic.keys())[16] + list(dfs_in_dic.keys())[17]
    print(first_four_keys)
    filtered_dict = {key: dfs_in_dic[key] for key in first_four_keys}
    print(filtered_dict.keys())
    ModelResults_ = []

    csv_filename = f'results_Ko10_Ki_{public_variables.Descriptor_}.csv'
    csv_filename_temp = f'results_Ko10_Ki_{public_variables.Descriptor_}_temp.csv' #if i break it early i still get some results
    
    for name, df in filtered_dict.items():
        print(name)
        fold_results = DNN_function(name, df, dfs_path)
        ModelResults_.append(fold_results)
        pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename_temp, index=False)
    
    pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename, index=False)

    return

if __name__ == "__main__":
    main(public_variables.dfs_descriptors_only_path_)
    main(public_variables.dfs_reduced_path_)
    main(public_variables.dfs_reduced_and_MD_path_)
    main(public_variables.dfs_MD_only_path_)
















# Load the data
# df = pd.read_csv(public_variables.dfs_descriptors_only_path_ / '1ns.csv')

# # Step 2: Split into features and target
# X = df.drop(columns=['PKI', 'mol_id', 'conformations (ns)'], axis=1, errors='ignore')
# y = df['PKI'].values.reshape(-1, 1)

# # Step 3: Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Step 4: Bin target values to handle imbalance
# bins = 5
# binned_y, bin_edges = pd.cut(y.flatten(), bins=bins, labels=False, retbins=True)
# dir_path = public_variables.base_path_ / 'code' / 'DNN' / 'train_val_loss' / 'test'
# dir_path.mkdir(parents=True, exist_ok=True)



# bin_counts = np.bincount(binned_y)  # Count samples per bin
# total_samples = len(binned_y)
# bin_weights = total_samples / (len(bin_counts) * bin_counts)  # Normalize weights

# exp_weights = bin_weights**0.5  # Adjust exponent for desired smoothing
# exp_weights /= exp_weights.max()  # Normalize to [0, 1]
# test_size = 0.2

# # Initialize variables
# outer_folds = 10
# outer_kf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
# r2_scores = []
# mse_scores = []

# hyperparameter_grid = [
#     {"learning_rate": 0.001, "hidden_layers": [64, 32, 16]},
#     {"learning_rate": 0.0005, "hidden_layers": [128, 64]},
#     {"learning_rate": 0.01, "hidden_layers": [64, 64, 32]},
#     {"learning_rate": 0.01, "hidden_layers": [128, 64, 32]},
#     # Add more combinations as needed
# ]

# # Store losses and predictions across folds
# all_predictions = []
# all_true_values = []
# all_train_losses = []
# all_val_losses = []

# for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X_scaled, binned_y)):
#     print(f"Outer Fold {fold + 1}")

#     # Split into train and test sets
#     X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]

#     # Inner split: Further split training data into 70% training, 10% validation
#     X_inner_train, X_val, y_inner_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.125, stratify=pd.cut(y_train.flatten(), bins=bins, labels=False),
#         random_state=42
#     )

#     # Convert to tensors for inner split
#     inner_train_dataset = TensorDataset(torch.tensor(X_inner_train, dtype=torch.float32),
#                                          torch.tensor(y_inner_train, dtype=torch.float32))
#     val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
#                                  torch.tensor(y_val, dtype=torch.float32))
#     train_loader = DataLoader(inner_train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     # Hyperparameter tuning
#     best_params = None
#     best_val_loss = float('inf')
#     for params in hyperparameter_grid:
#         model = FullyConnectedDNN_class2.FullyConnectedDNN2(input_size=X_train.shape[1], hidden_layers=params["hidden_layers"],dropout_rate=0.1)
#         optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
#         criterion = torch.nn.MSELoss()

#         # Train with validation to tune hyperparameters
#         train_losses, val_losses = model.train_with_validation(
#             train_loader, val_loader, num_epochs=100, optimizer=optimizer,
#             criterion=criterion, device="cuda" if torch.cuda.is_available() else "cpu"
#         )

#         # Track the best hyperparameters
#         if val_losses[-1] < best_val_loss:
#             best_val_loss = val_losses[-1]
#             best_params = params
#             best_train_losses = train_losses
#             best_val_losses = val_losses

#     print(f"Best params for fold {fold + 1}: {best_params}, Validation Loss: {best_val_loss:.4f}")

#     # Train final model on 80% training data with best hyperparameters
#     final_train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
#                                          torch.tensor(y_train, dtype=torch.float32))
#     final_train_loader = DataLoader(final_train_dataset, batch_size=32, shuffle=True)

#     final_model = FullyConnectedDNN_class2.FullyConnectedDNN2(input_size=X_train.shape[1], hidden_layers=best_params['hidden_layers'],dropout_rate=0.1)
#     final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
#     final_criterion = torch.nn.MSELoss()

#     final_model.train_without_validation(
#         final_train_loader, num_epochs=100, optimizer=final_optimizer,
#         criterion=final_criterion, device="cuda" if torch.cuda.is_available() else "cpu"
#     )

#     # Evaluate on test set
#     final_model.eval()
#     test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
#                                   torch.tensor(y_test, dtype=torch.float32))
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#     predictions = []
#     true_values = []

#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
#             outputs = final_model(inputs)
#             predictions.extend(outputs.cpu().numpy())
#             true_values.extend(targets.numpy())
#     # Append fold results
#     all_predictions.extend(predictions)
#     all_true_values.extend(true_values)
#     all_train_losses.append(best_train_losses)
#     all_val_losses.append(best_val_losses)
#     # Calculate metrics
#     mse = mean_squared_error(true_values, predictions)
#     r2 = r2_score(true_values, predictions)
#     mse_scores.append(mse)
#     r2_scores.append(r2)

#     print(f"Outer Fold {fold + 1} - Test MSE: {mse:.4f}, Test R²: {r2:.4f}")

# # Print overall metrics
# print(f"Average MSE: {np.mean(mse_scores):.4f}")
# print(f"Average R²: {np.mean(r2_scores):.4f}")

# # Plot Training and Validation Loss Curves
# plt.figure(figsize=(10, 6))
# for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
#     plt.plot(train_loss, label=f"Fold {fold + 1} - Train Loss")
#     plt.plot(val_loss, label=f"Fold {fold + 1} - Validation Loss", linestyle="--")

# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss Across Folds")
# plt.legend()
# plt.savefig(public_variables.base_path_ / 'code' / 'DNN' / 'train_loss_plots' / f'{name}_trainloss_all_folds.png')
# plt.close()

# for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_loss, label=f"Train Loss")
#     plt.plot(val_loss, label=f"Validation Loss", linestyle="--")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title(f"Fold {fold + 1} - Training and Validation Loss")
#     plt.legend()
#     plt.savefig(public_variables.base_path_ / 'code' / 'DNN' / 'train_loss_plots' / f'{name}_trainloss_fold_{fold + 1}.png')
#     plt.close()


# # Plot Predicted vs True pKi Values
# plt.figure(figsize=(8, 8))
# plt.scatter(all_true_values, all_predictions, alpha=0.6, edgecolor='k')
# plt.plot([min(all_true_values), max(all_true_values)],
#          [min(all_true_values), max(all_true_values)], color="red", linestyle="--", label="Ideal Fit")
# plt.xlabel("True pKi")
# plt.ylabel("Predicted pKi")
# plt.title("Predicted vs True pKi Across All Folds")
# plt.legend()
# plt.savefig(public_variables.base_path_ / 'code' / 'DNN' / '1ns_predtrue_DNN3.png')
# plt.close()