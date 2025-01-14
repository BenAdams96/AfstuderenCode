from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import randomForest_read_in_models
import csv_to_dataframes

from Afstuderen0.Afstuderen.code.A_RF_Class import RandomForestModel
import public_variables

from sklearn.preprocessing import StandardScaler
from csv_to_dataframes import csvfiles_to_dfs
import csv_to_dictionary
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
from typing import Dict
import numpy as np
from pathlib import Path

import pandas as pd
import math
import re
import os

def Kfold_Cross_Validation_incl_grouped(dfs_in_dic, hyperparameter_grid, kfold_, scoring_):
    
    
    columns_ = ["mol_id", "mean_test_score", "std_test_score", "params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_)
    models = {}

    for name, df in dfs_in_dic.items():
        print(name)

        targets = df['PKI']
        unique_mol_ids = df['mol_id'].unique()
        
        kf = KFold(n_splits=kfold_, shuffle=True, random_state=1)
        splits = list(kf.split(unique_mol_ids))
        
        custom_splits = []
        for train_mol_indices, test_mol_indices in kf.split(unique_mol_ids):
            # Get actual molecule IDs for train and test sets
            train_mols = unique_mol_ids[train_mol_indices]
            test_mols = unique_mol_ids[test_mol_indices]
            
            # Map back to the full dataset indices (6000 rows)
            train_indices = df[df['mol_id'].isin(train_mols)].index
            test_indices = df[df['mol_id'].isin(test_mols)].index
            
            # Append as a tuple of arrays
            custom_splits.append((train_indices, test_indices))
        # print('custom splits')
        # print(custom_splits[1][0][0:40])
        # print(custom_splits[1][1][0:40])
        # print(targets[0:40])
        # Initial model and grid search outside the loop
        rf_model = RandomForestModel(n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt')
        rf_model.model.random_state = 42
        df = df.drop(columns=['mol_id','PKI','conformations (ns)'], axis=1, errors='ignore')
        
        grid_search = rf_model.hyperparameter_tuning(df,targets,hyperparameter_grid,cv=custom_splits,scoring_=scoring_)
        
        df_results = pd.DataFrame(grid_search.cv_results_) #what results? of cv. so the 10 results and 8 rows (each row was a set of hyperparameters)
        
        #grid_search.best_index_ = the index that is best of the x amount of hyperparameter combinations
        result = df_results.loc[grid_search.best_index_, columns_[1:]] #1: because we dont want to include 'time'
        #df_results = dataframe with 'hyperparametercombinations' amount of rows, and each row has mean_test_score and split_test_scores etc
        #result = of the best combination of hyperparameters. mean_test_score, std, params, split 0 to 4
        #make it so that the scores are always positive
        result["mean_test_score"] = abs(result['mean_test_score']) 
        for i in range(kfold_):
            result[f'split{i}_test_score'] = abs(result[f'split{i}_test_score'])

        result_df_row = result.to_frame().T
        result_df_row['mol_id'] = name #add a column of the '1ns' etc
        result_df_row = result_df_row[columns_] #change the order to the one specified above

        #make 1 big dataframe for all the '0ns' '1ns' 'rdkit_min' etc
        ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)
        models[name] = rf_model
    return models, ModelResults_

def strip_dataframes(dataframes,feature_index_list_of_lists):
    stripped_dfs = []
    for idx,df in enumerate(dataframes):
        df_stripped = strip_dataframe(df,feature_index_list_of_lists[idx])
        stripped_dfs.append(df_stripped)
    return stripped_dfs

def strip_dataframe(dataframe,feature_index_list):
    df = dataframe.dropna()
    df.reset_index(drop=True, inplace=True)
    sorted_feature_index_list = sorted(feature_index_list, key=int)
    #print(sorted_feature_index_list)
    stripped_df = df[['mol_id', 'PKI'] + sorted_feature_index_list]
    return stripped_df

def save_dataframes_to_csv(dic_with_dfs,save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in dic_with_dfs.items():
        print(f"save dataframe: {name}")
        df.to_csv(save_path / f'{name}.csv', index=False)








def bin_pki_values(y, num_bins=5):
    """
    Bin the pKi values into discrete bins for stratified splitting.
    Parameters:
        y (array-like): Target variable (pKi values).
        num_bins (int): Number of bins to stratify.
    Returns:
        binned_y: Binned version of y.
    """
    bins = np.linspace(y.min(), y.max(), num_bins + 1)  # Define bin edges
    binned_y = pd.cut(y, bins, right=True, include_lowest=True, labels=False)  # Include both sides
    return bins, binned_y

def visualize_folds_distribution(y, fold_indices, title="pKi Distribution Across Folds"):
    """
    Visualize the distribution of pKi values across folds.
    Parameters:
        y (array-like): Target variable (pKi values).
        fold_indices (list of tuples): List of (train_idx, test_idx) for each fold.
    """
    plt.figure(figsize=(10, 6))
    for fold_num, (train_idx, test_idx) in enumerate(fold_indices, 1):
        train_values = y[train_idx]
        test_values = y[test_idx]
        plt.hist(train_values, bins=10, alpha=0.5, label=f"Fold {fold_num} Train", histtype='step')
        plt.hist(test_values, bins=10, alpha=0.5, label=f"Fold {fold_num} Test", histtype='stepfilled')
    
    plt.xlabel("pKi Values")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.savefig('save.png')

def plot_predicted_vs_real_pKi(y_true, y_pred, number):
    """
    Plots predicted pKi values against real pKi values for model evaluation.

    Parameters:
        y_true (array-like): The true pKi values.
        y_pred (array-like): The predicted pKi values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='k', s=80)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')  # Ideal line
    plt.xlabel('Real pKi Values')
    plt.ylabel('Predicted pKi Values')
    plt.title('Predicted vs Real pKi Values')
    plt.grid(True)
    plt.savefig(f'true_vs_predict_{number}.png')

def bin_pki_values2(pki_values, bins):
    return pd.cut(pki_values, bins=bins, labels=False, include_lowest=True)

def calculate_accuracy_by_bin(all_real_pki_series, all_predicted_pki_series, pki_bins):
    # Ensure the lengths of the real and predicted arrays match
    if len(all_real_pki_series) != len(all_predicted_pki_series):
        raise ValueError("The length of binned_real and binned_predicted must be the same.")
    
    # Define the bin labels for each bin
    bin_labels = list(range(len(pki_bins) - 1))  # Labels like 0, 1, 2, etc.
    print(pki_bins)
    print(bin_labels)
    # Bin the real pKi values (include upper boundary for the last bin)
    binned_real_labels = pd.cut(all_real_pki_series, bins=pki_bins, labels=bin_labels, right=True, include_lowest=True)

    # Handle NaN values after binning (values outside the bin range)
    if binned_real_labels.isna().any():
        raise ValueError("Some values in binned_real are outside the pki_bins range.")
    
    # Convert the labels to integers
    binned_real_labels = binned_real_labels.astype(int)
    
    # Initialize the dictionary to store R² scores for each bin
    accuracy_by_bin = {}
    
    # Loop through each bin
    for bin_idx in range(len(pki_bins) - 1):
        print(bin_idx)
        # Mask to select values in the current bin
        bin_mask = binned_real_labels == bin_idx #mask of all that are in binned_real_labels but True if they are in this

        # Filter real and predicted values based on the bin mask
        true_within_bin = all_real_pki_series[bin_mask]
        pred_within_bin = all_predicted_pki_series[bin_mask]
        print(true_within_bin)
        print(pred_within_bin)
        if not true_within_bin.empty and not pred_within_bin.empty:
            bin_r2 = r2_score(true_within_bin, pred_within_bin)
        else:
            bin_r2 = None  # Set R² to None if no values in the bin
        print(bin_r2)
        # Store the R² score for the bin
        accuracy_by_bin[f"bin_{pki_bins[bin_idx]}_{pki_bins[bin_idx+1]}"] = bin_r2
    
    return accuracy_by_bin

def true_and_predicted_tocsv(df,all_true_pki_pdseries, all_predicted_pki_pdseries, dfs_path):
    save_path = dfs_path / public_variables.Modelresults_folder_ / 'true_vs_prediction' / 'save_it.csv'

    # Ensure the parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "True_pKi": all_true_pki_pdseries,
        "Predicted_pKi": all_predicted_pki_pdseries
    })
    df.to_csv(save_path, index=False)
    return

def nested_cross_validation(name, df, dfs_path, outer_folds=5, inner_folds=5):
    """
    Perform nested cross-validation with stratified outer folds.
    Parameters:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        outer_folds (int): Number of outer folds.
        inner_folds (int): Number of inner folds.
    """
    print("Nested Cross Validation")

    target_column = 'PKI'
    X = df.drop(columns=[target_column, 'mol_id', 'conformations (ns)'], axis=1, errors='ignore')
    
    y = df[target_column]  # Target (pKi values)

    # Stratified outer loop
    bins, binned_y = bin_pki_values(y, num_bins=5)  # Bin pKi values for stratification, still a dataframe: all molecules with an index of which bin they are in
    unique, counts = np.unique(binned_y, return_counts=True)

    # Group by mol_id
    grouped_df = df.groupby('mol_id').first()  # One row per molecule, so group by mol_id
    # bins2, grouped_binned_y = bin_pki_values(grouped_df[target_column], num_bins=5)
    grouped_binned_y = binned_y.groupby(df['mol_id']).first()
    groups_molid = grouped_df.index  # Index([1,2,3,4,6]) array of all mol_id
    
    fold_results = []
    custom_outer_splits = []
    all_true_pki_series = pd.Series(dtype=float)
    all_predicted_pki_series = pd.Series(dtype=float)
    outer_cv = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(grouped_df, grouped_binned_y, groups=groups_molid)):
        # print(f'Outer fold: {outer_fold}')
        
        # Map fold indices back to the original dataframe
        train_mols = grouped_df.iloc[train_idx].index  # Train molecule IDs
        test_mols = grouped_df.iloc[test_idx].index  # Test molecule IDs
        
        train_idx_full = df[df['mol_id'].isin(train_mols)].index  # All rows for train molecules
        test_idx_full = df[df['mol_id'].isin(test_mols)].index  # All rows for test molecules
        custom_outer_splits.append((train_idx_full, test_idx_full))

        # Split data
        X_train, X_test = X.loc[train_idx_full], X.loc[test_idx_full]
        y_train, y_test = y.loc[train_idx_full], y.loc[test_idx_full]

        groups_train = df.loc[X_train.index, 'mol_id']  # Train set molecule IDs
        groups_test = df.loc[X_test.index, 'mol_id']  # Test set molecule IDs

        # Bin y_train for stratification in inner loop
        binned_y_train = binned_y.loc[X_train.index] #df with idx of all molecules/conformations and the bin it belongs to.

        best_model = None
        best_score = -np.inf
        
        # Inner loop: StratifiedGroupKFold
        inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=42)
        custom_inner_splits = []
        all_real_pki_inner = []
        all_predicted_pki_inner = []
        # Group `X_train` by `mol_id`
        grouped_X_train = X_train.groupby(groups_train).first()  # One row per molecule
        grouped_binned_y_train = binned_y_train.groupby(groups_train).first()  # Match grouping
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(grouped_X_train, grouped_binned_y_train, groups=grouped_X_train.index)):
            # print(f'  Inner fold: {inner_fold}')
            
            # Map fold indices back to the original molecule IDs
            inner_train_mols = grouped_X_train.iloc[inner_train_idx].index
            inner_val_mols = grouped_X_train.iloc[inner_val_idx].index

            # Expand molecule IDs to full conformations in `X_train`
            inner_train_idx_full = df[df['mol_id'].isin(inner_train_mols)].index.intersection(X_train.index)
            inner_val_idx_full = df[df['mol_id'].isin(inner_val_mols)].index.intersection(X_train.index)
            custom_inner_splits.append((inner_train_idx_full, inner_val_idx_full))
            # Split inner train/validation data
            X_train_inner, X_val = X_train.loc[inner_train_idx_full], X_train.loc[inner_val_idx_full]
            y_train_inner, y_val = y_train.loc[inner_train_idx_full], y_train.loc[inner_val_idx_full]

        rf_model = RandomForestModel(n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt')
        rf_model.model.random_state = 42
        
        grid_search = rf_model.hyperparameter_tuning(df, y,public_variables.hyperparameter_grid_RF,cv=custom_inner_splits,scoring_='r2')
        df_results = pd.DataFrame(grid_search.cv_results_)
        # print(df_results)
        #build a "final" model for this train test split.
        rf_model = RandomForestModel(n_trees=grid_search.best_params_['n_estimators'],
                                 max_depth=grid_search.best_params_['max_depth'],
                                 min_samples_split=grid_search.best_params_['min_samples_split'],
                                 max_features=grid_search.best_params_['max_features'])
        rf_model.model.random_state = 42
        rf_model.model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        r2_score = rf_model.model.score(X_test, y_test)
        print(f'R2 score for outer fold {outer_fold}: {r2_score}')
        fold_results.append(r2_score) #append r2_score to the outer_fold scores
       
        all_true_pki_series = pd.concat([all_true_pki_series, y_test]).sort_index() #pd Series of all true pki values
        all_predicted_pki_series = pd.concat([all_predicted_pki_series, pd.Series(y_pred, index=y_test.index)]).sort_index()

    #save the true and predicted values in a csv file for later analysis
    mol_id_and_ns = df[['mol_id'] + (['conformations (ns)'] if 'conformations (ns)' in df else [])]
    print(mol_id_and_ns)
    true_pred_pki_df = pd.DataFrame({
        'mol_id': mol_id_and_ns['mol_id'],
        'True_pKi': all_true_pki_series,
        'Predicted_pKi': all_predicted_pki_series
    })

    # Add 'conformations (ns)' only if it exists in mol_id_and_ns
    if 'conformations (ns)' in mol_id_and_ns:
        true_pred_pki_df.insert(1, 'conformations (ns)', mol_id_and_ns['conformations (ns)'])
    # true_and_predicted_tocsv(all_true_pki_series, all_predicted_pki_series, dfs_path)
    save_path = dfs_path / public_variables.Modelresults_folder_ / 'true_vs_prediction' / f'{name}.csv'

    # Ensure the parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    true_pred_pki_df.to_csv(save_path, index=False)

    # Store the results for the fold
    data = pd.Series(fold_results)
    fold_results = {
        "mol_id": name,
        "mean_test_score": data.mean(),
        "std_test_score": data.std(),
        "params": grid_search.best_params_,
        **{f"split{split_idx}_test_score": fold_results[split_idx] for split_idx in range(outer_folds)},  # Loop to add splits
    }

    return fold_results


def main(dfs_path = public_variables.dfs_descriptors_only_path_):

    #create folder for storing the models and results from them
    Modelresults_path = dfs_path / public_variables.Modelresults_folder_ #CHECK: if 'RF' is in public_variables or something else perhaps
    Modelresults_path.mkdir(parents=True, exist_ok=True)

    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic(dfs_path, exclude_files=['conformations_1000.csv','conformations_1000_molid.csv','conformations_500.csv','conformations_200.csv','conformations_100.csv','conformations_50.csv','initial_dataframe.csv','initial_dataframes_best.csv','MD_output.csv']) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys())) #RDKIT first
    dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic} #order
    print(dfs_in_dic.keys())

    dic_models = {}

    columns_ = ["mol_id", "mean_test_score", "std_test_score", "params"]
    ModelResults_ = pd.DataFrame(columns=columns_)

    outer_folds = 5
    inner_folds = 5
    scoring = 'r2'
    
    # #NOTE for random testing
    # df = dfs_in_dic['conformations_10']
    # fold_results = nested_cross_validation('test', df, dfs_path,outer_folds, inner_folds)

    for name, df in dfs_in_dic.items():
        print(name)
        fold_results = nested_cross_validation(name, df, dfs_path, outer_folds, inner_folds)
        row_results = pd.DataFrame([fold_results])
        ModelResults_ = pd.concat([ModelResults_, row_results], ignore_index=True)

    csv_filename = f'results_Ko{outer_folds}_Ki{inner_folds}_{scoring}_{public_variables.RDKIT_descriptors_}.csv'
    ModelResults_.to_csv(Modelresults_path / csv_filename, index=False)


    # # plot_predicted_vs_real_pKi(y_test, y_pred,5) #TODO: create an apart python file for this + add accuracy by bin to this
    # accuracy_by_bin = calculate_accuracy_by_bin(all_true_pki_series, all_predicted_pki_series, [4,10]) #[4,10] r2 score is 0.737, but when [4,6,8,10] all very bad
    return

if __name__ == "__main__":
    main(public_variables.dfs_descriptors_only_path_)
    main(public_variables.dfs_reduced_path_)
    main(public_variables.dfs_PCA_path)
    main(public_variables.dfs_reduced_and_MD_path_)
    main(public_variables.dfs_MD_only_path_)