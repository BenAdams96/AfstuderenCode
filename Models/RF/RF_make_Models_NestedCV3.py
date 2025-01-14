from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from global_files import csv_to_dataframes

from Models.RF.RF_Class import RandomForestModel
from global_files import public_variables
from Models.RF.BaseModel_Classes import RandomForestModel2, XGBoostModel2, SVRModel2

from sklearn.preprocessing import StandardScaler

from global_files import csv_to_dictionary
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

def nested_cross_validation(name, df, dfs_path, outer_folds=10, inner_folds=5):
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
    print(bins)
    print(counts)
    # Group by mol_id
    grouped_df = df.groupby('mol_id').first()  # One row per molecule, so group by mol_id
    # bins2, grouped_binned_y = bin_pki_values(grouped_df[target_column], num_bins=5)
    grouped_binned_y = bin_pki_values(grouped_df[target_column], num_bins=5)[1]
    groups_molid = grouped_df.index  # Index([1,2,3,4,6]) array of all mol_id
    
    fold_results = []
    custom_outer_splits = []
    all_true_pki_series = pd.Series(dtype=float)
    all_predicted_pki_series = pd.Series(dtype=float)
    outer_cv = StratifiedGroupKFold(n_splits=outer_folds,shuffle=True, random_state=10)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(grouped_df, grouped_binned_y, groups=groups_molid)):
        print(f'Outer fold: {outer_fold}')
        
        # Map fold indices back to the original dataframe
        train_mols = grouped_df.iloc[train_idx].index  # Train molecule IDs
        test_mols = grouped_df.iloc[test_idx].index  # Test molecule IDs
        
        train_idx_full = df[df['mol_id'].isin(train_mols)].index  # All rows for train molecules
        test_idx_full = df[df['mol_id'].isin(test_mols)].index  # All rows for test molecules
        custom_outer_splits.append((train_idx_full, test_idx_full))
        
        unique, counts = np.unique(binned_y.iloc[test_idx_full], return_counts=True)
        # print(unique)
        # print(counts)
        # print(np.sum(counts))

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
        inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=10)
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
            unique, counts = np.unique(binned_y.iloc[inner_val_idx_full], return_counts=True)
            # print(unique)
            # print(counts)
            # print(np.sum(counts))
            custom_inner_splits.append((inner_train_idx_full, inner_val_idx_full))
            # Split inner train/validation data
            X_train_inner, X_val = X_train.loc[inner_train_idx_full], X_train.loc[inner_val_idx_full]
            y_train_inner, y_val = y_train.loc[inner_train_idx_full], y_train.loc[inner_val_idx_full]

        rf_model = RandomForestRegressor(n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt', random_state=42)

        grid_search = rf_model.hyperparameter_tuning(X, y,public_variables.hyperparameter_grid_RF,cv=custom_inner_splits,scoring_='r2')
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
    true_pred_pki_df = pd.DataFrame({
        'mol_id': mol_id_and_ns['mol_id'],
        'True_pKi': all_true_pki_series,
        'Predicted_pKi': all_predicted_pki_series
    })

    # Add 'conformations (ns)' only if it exists in mol_id_and_ns
    if 'conformations (ns)' in mol_id_and_ns:
        true_pred_pki_df.insert(1, 'conformations (ns)', mol_id_and_ns['conformations (ns)'])

    #make sure the sorting is nice, for better viewing
    sort_columns = ['mol_id'] + (['conformations (ns)'] if 'conformations (ns)' in true_pred_pki_df else [])
    true_pred_pki_df = true_pred_pki_df.sort_values(by=sort_columns).reset_index(drop=True)
    save_path = dfs_path / 'ModelResults_RF' / 'true_vs_prediction' / f'{name}.csv'

    # Ensure the parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    true_pred_pki_df.to_csv(save_path, index=False)

    # Store the results for the fold
    data = pd.Series(fold_results)
    print(f'R2 score: {data.mean()}')
    print(f'R2 std: {data.std()}')
    fold_results = {
        "mol_id": name,
        "mean_test_score": data.mean(),
        "std_test_score": data.std(),
        "params": grid_search.best_params_,
        **{f"split{split_idx}_test_score": fold_results[split_idx] for split_idx in range(outer_folds)},  # Loop to add splits
    }

    return fold_results


def main(dfs_path = public_variables.dfs_descriptors_only_path_):
    exclude_files_clusters = [
    'clustering_target10.0%_cluster1.csv', 'clustering_target10.0%_cluster2.csv',
    'clustering_target10.0%_cluster3.csv', 'clustering_target10.0%_cluster4.csv',
    'clustering_target10.0%_cluster5.csv', 'clustering_target10.0%_cluster6.csv',
    'clustering_target10.0%_cluster7.csv', 'clustering_target10.0%_cluster8.csv',
    'clustering_target10.0%_cluster9.csv', 'clustering_target10.0%_cluster10.csv',
    'clustering_target20.0%_cluster1.csv', 'clustering_target20.0%_cluster2.csv',
    'clustering_target20.0%_cluster3.csv', 'clustering_target20.0%_cluster4.csv',
    'clustering_target20.0%_cluster5.csv', 'clustering_target20.0%_cluster6.csv',
    'clustering_target20.0%_cluster7.csv', 'clustering_target20.0%_cluster8.csv',
    'clustering_target20.0%_cluster9.csv', 'clustering_target20.0%_cluster10.csv',
    'clustering_target30.0%_cluster1.csv', 'clustering_target30.0%_cluster2.csv',
    'clustering_target30.0%_cluster3.csv', 'clustering_target30.0%_cluster4.csv',
    'clustering_target30.0%_cluster5.csv', 'clustering_target30.0%_cluster6.csv',
    'clustering_target30.0%_cluster7.csv', 'clustering_target30.0%_cluster8.csv',
    'clustering_target30.0%_cluster9.csv', 'clustering_target30.0%_cluster10.csv',
    'clustering_target40.0%_cluster1.csv', 'clustering_target40.0%_cluster2.csv',
    'clustering_target40.0%_cluster3.csv', 'clustering_target40.0%_cluster4.csv',
    'clustering_target40.0%_cluster5.csv', 'clustering_target40.0%_cluster6.csv',
    'clustering_target40.0%_cluster7.csv', 'clustering_target40.0%_cluster8.csv',
    'clustering_target40.0%_cluster9.csv', 'clustering_target40.0%_cluster10.csv',
    'clustering_target50.0%_cluster1.csv', 'clustering_target50.0%_cluster2.csv',
    'clustering_target50.0%_cluster3.csv', 'clustering_target50.0%_cluster4.csv',
    'clustering_target50.0%_cluster5.csv', 'clustering_target50.0%_cluster6.csv',
    'clustering_target50.0%_cluster7.csv', 'clustering_target50.0%_cluster8.csv',
    'clustering_target50.0%_cluster9.csv', 'clustering_target50.0%_cluster10.csv'
    ]

    #create folder for storing the models and results from them
    model_ = 'RF'
    Modelresults_path = dfs_path / f'ModelResults_{model_}' #CHECK: if 'RF' is in public_variables or something else perhaps
    Modelresults_path.mkdir(parents=True, exist_ok=True)
    only_clusters = False
    if only_clusters == True:
        exclude_files = ['0ns.csv','1ns.csv','2ns.csv','3ns.csv','4ns.csv','5ns.csv','6ns.csv','7ns.csv','8ns.csv','9ns.csv','10ns.csv','conformations_1000.csv','conformations_1000_molid.csv','conformations_500.csv','conformations_200.csv','conformations_100.csv','conformations_50.csv','initial_dataframe.csv','initial_dataframes_best.csv','MD_output.csv','conformations_10.csv','conformations_20.csv','minimized_conformations_10.csv','stable_conformations.csv'] #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary]
    else:
        exclude_files=['conformations_1000.csv','conformations_1000_molid.csv','conformations_500.csv','conformations_200.csv','conformations_100.csv','conformations_50.csv','initial_dataframe.csv','initial_dataframes_best.csv','MD_output.csv'] + exclude_files_clusters #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic(dfs_path, exclude_files=exclude_files)
    print(dfs_in_dic.keys())
    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys())) #RDKIT first
    dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic} #order
    print(dfs_in_dic.keys())
    first_four_keys = list(dfs_in_dic.keys())[-4:]
    filtered_dict = {key: dfs_in_dic[key] for key in first_four_keys}
    print(filtered_dict.keys())


    ModelResults_ = []

    outer_folds = 10
    inner_folds = 5
    scoring = 'r2'
    
    # #NOTE for random testing
    # df = dfs_in_dic['conformations_10']
    # df = dfs_in_dic['99ns']
    # simplified_df = df.iloc[:, :4]
    # Assign values in increments of 1 for every 10 rows
    # df['mol_id'] = df.index + 1
    
    # fold_results = nested_cross_validation('test', simplified_df, dfs_path,outer_folds, inner_folds)
    csv_filename = f'results_Ko{outer_folds}_Ki{inner_folds}_{scoring}_{public_variables.Descriptor_}.csv'
    csv_filename_temp = f'results_Ko{outer_folds}_Ki{inner_folds}_{scoring}_{public_variables.Descriptor_}_temp.csv'
    csv_filename_cluster = f'results_Ko{outer_folds}_Ki{inner_folds}_{scoring}_{public_variables.Descriptor_}_clusters.csv'
    csv_filename_temp_cluster = f'results_Ko{outer_folds}_Ki{inner_folds}_{scoring}_{public_variables.Descriptor_}_temp_clusters.csv' #if i break it early i still get some results
    for name, df in dfs_in_dic.items():
        print(name)
        fold_results = nested_cross_validation(name, df, dfs_path, outer_folds, inner_folds)
        ModelResults_.append(fold_results)
        if only_clusters == True:
            pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename_temp_cluster, index=False)
        else:
            pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename_temp, index=False)
    if only_clusters == True:
        pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename_cluster, index=False)
    else:
        pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename, index=False)
    return

if __name__ == "__main__":
    # main(public_variables.dataframes_master_ / '2D')
    # main(public_variables.dfs_descriptors_only_path_)
    # main(public_variables.dfs_descriptors_only_path_)
    # main(public_variables.dfs_PCA_path)
    # dataframes_master_ = public_variables.base_path_ / Path(f'dataframes_GSK3_WHIM')
    # # main(dataframes_master_ / 'reduced_t0.85')
    # # main(dataframes_master_ / 'reduced_t0.85_MD')
    # main(dataframes_master_ / 'MD only')

    # main(public_variables.dfs_reduced_path_)
    # main(public_variables.dfs_reduced_and_MD_path_)
    main(public_variables.dfs_MD_only_path_)

    