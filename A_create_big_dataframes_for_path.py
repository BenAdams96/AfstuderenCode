from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

import randomForest_read_in_models
import csv_to_dataframes
import Afstuderen0.Afstuderen.code.A_randomForest_Class as A_randomForest_Class
from Afstuderen0.Afstuderen.code.A_randomForest_Class import RandomForestModel
import public_variables

from sklearn.preprocessing import StandardScaler
from csv_to_dataframes import csvfiles_to_dfs
import csv_to_dictionary

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

def append_modelResults_to_csv(ModelResults, modelresults_path, csv_filename, concat_direction):
    # Check if the CSV file exists
    if modelresults_path.exists():
        # Load the CSV file to check if 'combined' exists in the 'id' column
        csv_path = modelresults_path / csv_filename
        if csv_path.exists():
            existing_df = pd.read_csv(modelresults_path / csv_filename)
            
            # If 'combined' exists in the 'id' column, do not append the row
            if 'id' in existing_df.columns and f'concat_{concat_direction}' in existing_df['id'].values:
                print("Row with 'combined' in 'id' column already exists. No new row added. changed tho")
                existing_df = existing_df[existing_df['id'] != f'concat_{concat_direction}']
                # Append the new results to the updated DataFrame
                updated_df = pd.concat([existing_df, ModelResults])
                updated_df.to_csv(csv_path, index=False)
                return
            else:
                # Append the row since 'combined' does not exist
                ModelResults.to_csv(modelresults_path / csv_filename, mode='a', header=False, index=False)
                print("Row appended to CSV.")
        else:
            print('else: modelresults.tocsv modelresults_path / csv_filename')
            ModelResults.to_csv(modelresults_path / csv_filename, index=False)
    else:
        print('cant append combined results, file doesnt exists')

def Kfold_Cross_Validation_bigModel(combined_dfs, hyperparameter_grid, kfold_, scoring_): #add targets
    print("Kfold_Cross_Validation_bigModel")
    
    columns_ = ["id","mean_test_score","std_test_score","params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_) #NOTE needs to be a pd dataframe

    # Extract features (X) and targets (y) from the combined DataFrame
    y_train = combined_dfs['PKI']  # Target column
    X_train = combined_dfs.drop(columns=['mol_id','PKI'])  # Features are all columns except target
    
    # Define the KFold cross-validator
    kf = KFold(n_splits=kfold_, shuffle=True, random_state=1)
    
    # Initialize the RandomForest model
    rf_model = RandomForestModel(n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt')
    rf_model.model.random_state = 42  # Set the random state for reproducibility
    
    # Perform hyperparameter tuning with GridSearchCV
    grid_search = rf_model.hyperparameter_tuning(X_train, y_train, hyperparameter_grid, cv=kf, scoring_=scoring_)

    # Convert the cross-validation results into a DataFrame
    df_results = pd.DataFrame(grid_search.cv_results_)
    
    # Get the results of the best hyperparameter combination
    result = df_results.loc[grid_search.best_index_, columns_[1:]]  # Exclude the 'id' column?
    
    # Ensure all test scores are positive
    result["mean_test_score"] = abs(result["mean_test_score"])
    for i in range(kfold_):
        result[f'split{i}_test_score'] = abs(result[f'split{i}_test_score'])

     # Create a DataFrame row with the results and reorder the columns
    result_df_row = result.to_frame().T
    result_df_row['id'] = 'concat'
    result_df_row = result_df_row[columns_]  # Ensure the column order is correct

    # Append the results to the main DataFrame
    ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)

    print('ending kfold cross validation')
    # Return the trained model and the cross-validation results DataFrame
    return rf_model, ModelResults_

def Kfold_Cross_Validation_bigModel_grouped(combined_dfs, hyperparameter_grid, kfold_, scoring_, concat_direction): #add targets
    print("Kfold_Cross_Validation_bigModel grouped")
    print(concat_direction)
    
    columns_ = ["id","mean_test_score","std_test_score","params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_) #NOTE needs to be a pd dataframe
    
    # Extract features (X) and targets (y) from the combined DataFrame
    y_train = combined_dfs['PKI']  # Target column
    X_train = combined_dfs.drop(columns=['mol_id','PKI'])  # Features are all columns except target
    groups = combined_dfs['mol_id']
    
    # Define the GroupKFold cross-validator
    group_kf = GroupKFold(n_splits=kfold_)
    
    # Initialize the RandomForest model
    rf_model = RandomForestModel(n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt')
    rf_model.model.random_state = 42  #Set the random state for reproducibility
    
    # Perform hyperparameter tuning with GridSearchCV, using GroupKFold as cross-validator
    grid_search = rf_model.hyperparameter_tuning(X_train, y_train, hyperparameter_grid, cv=group_kf.split(X_train, y_train, groups=groups), scoring_=scoring_)

    # Convert the cross-validation results into a DataFrame
    df_results = pd.DataFrame(grid_search.cv_results_)
    
    # Get the results of the best hyperparameter combination
    result = df_results.loc[grid_search.best_index_, columns_[1:]]  # Exclude the 'id' column?
    
    # Ensure all test scores are positive
    result["mean_test_score"] = abs(result["mean_test_score"])
    for i in range(kfold_):
        result[f'split{i}_test_score'] = abs(result[f'split{i}_test_score'])

     # Create a DataFrame row with the results and reorder the columns
    result_df_row = result.to_frame().T
    result_df_row['id'] = f'concat_{concat_direction}'
    result_df_row = result_df_row[columns_]  # Ensure the column order is correct

    # Append the results to the main DataFrame
    ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)
    
    print('ending kfold cross validation')
    # Return the trained model and the cross-validation results DataFrame
    return rf_model, ModelResults_


def natural_sort_key(file_name):
    # Extract the numeric part of the filename (e.g., '0ns.csv' -> 0, '10ns.csv' -> 10)
    return int(re.search(r'(\d+)', file_name.stem).group(1))

def combine_csv_by_molecule_order_vertically(folder_path: Path, exclude_files: list = ['output.csv', 'rdkit_min.csv', 'concat_ver.csv', '0ns.csv', 'concat_hor.csv']):
    # Get list of CSV files in the folder, excluding specified files and sorting them naturally
    csv_files = sorted([file for file in folder_path.glob('*.csv') if file.name not in exclude_files], key=natural_sort_key)
    
    # Read all CSV files into a list of DataFrames
    dataframes = [pd.read_csv(file) for file in csv_files]

    # Stack the dataframes by molecule order (row-wise combination)
    combined_df = pd.concat([pd.concat([df.iloc[[i]] for df in dataframes], axis=0, ignore_index=True)
                             for i in range(len(dataframes[0]))], axis=0, ignore_index=True)
    
    # Save combined DataFrame as 'combined_v.csv'
    output_file = folder_path / 'concat_ver.csv'
    combined_df.to_csv(output_file, index=False)
    
    return combined_df

def combine_csv_by_molecule_order_horizontally(folder_path: Path, exclude_files: list = ['output.csv', 'rdkit_min.csv', 'concat_ver.csv', '0ns.csv', 'concat_hor.csv']):
    # Get list of CSV files in the folder, excluding specified files and sorting them naturally
    csv_files = sorted([file for file in folder_path.glob('*.csv') if file.name not in exclude_files], key=natural_sort_key)
    
    reference_df = pd.read_csv(csv_files[0])
    pki_mol_id_cols = reference_df[['mol_id','PKI']]

    # Read all CSV files into a list of DataFrames
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        df = df.drop(columns=['PKI', 'mol_id'], errors='ignore')  # Remove 'PKI' and 'mol_id' columns
        dataframes.append(df)

    # Stack the dataframes by molecule order (row-wise combination)
    combined_df = pd.concat(dataframes, axis=1, ignore_index=True)
    combined_df = pd.concat([pki_mol_id_cols, combined_df], axis=1)
    # Save combined DataFrame as 'combined_h.csv'
    output_file = folder_path / 'concat_hor.csv'
    combined_df.to_csv(output_file, index=False)
    return combined_df

def reduce_conformations(df, interval=1):
    """
    Reduces the number of conformations per molecule in the dataframe
    by selecting only specific conformations at given intervals, excluding 0.
    
    Parameters:
        df (pd.DataFrame): The large dataframe containing all conformations.
        interval (float): The desired interval for selection, default is 1ns.
    
    Returns:
        pd.DataFrame: A reduced dataframe with only the specified conformations per molecule.
    """
    # Define the target conformations, starting from the first interval, excluding 0
    target_conformations = [round(i * interval, 2) for i in range(1, int(10 / interval) + 1)]
    
    # Filter the dataframe to only include rows with conformations in target_conformations
    reduced_df = df[df['conformations (ns)'].isin(target_conformations)].copy(False)
    
    return reduced_df

def main(dfs_path = public_variables.dfs_descriptors_only_path_):  ###set as default, but can always change it to something else.`
    
    if not dfs_path.exists():
        print(f"Error: The path '{dfs_path}' does not exist.")
        return
    
    timeinterval = [1,0.5,0.2,0.1]
    initial_df = pd.read_csv(public_variables.initial_dataframe)
    print(initial_df)

    for t in timeinterval:
        print(t)
        reduced_dataframe = reduce_conformations(initial_df, interval=t)
        reduced_dataframe.to_csv(dfs_path / f'conformations_{int(10/t)}.csv', index=False)
    # combined_df_v = combine_csv_by_molecule_order_vertically(dfs_path) #creates 'concat_ver.csv'
    # combined_df_h = combine_csv_by_molecule_order_horizontally(dfs_path) #creates 'concat_hor.csv'
    
    # hyperparameter_grid = public_variables.hyperparameter_grid_
    # RDKIT_descriptors = public_variables.RDKIT_descriptors_
    # kfold = 10
    # scoring = ('r2','R-squared (R²)')

    # rf_model, ModelResults = Kfold_Cross_Validation_bigModel_grouped(combined_df_v, hyperparameter_grid, kfold, scoring[0], concat_direction='ver')
    
    # modelresults_path_ = dfs_path / public_variables.Modelresults_folder_
    # modelresults_combined_path_ = dfs_path / public_variables.Modelresults_combined_folder_
    # modelresults_combined_path_.mkdir(parents=True, exist_ok=True)
    # csv_filename = f'results_K{kfold}_{scoring[1]}_{RDKIT_descriptors}.csv'
    # append_modelResults_to_csv(ModelResults, modelresults_path_, csv_filename, concat_direction='ver') #TODO: or dont append, but use the two csv files when creating the figures
    # csv_filename = f'results_K{kfold}_{scoring[1]}_{RDKIT_descriptors}_ver.csv'
    # ModelResults.to_csv(modelresults_combined_path_ / csv_filename, index=False)


    # rf_model, ModelResults = Kfold_Cross_Validation_bigModel_grouped(combined_df_h, hyperparameter_grid, kfold, scoring[0], concat_direction='hor')

    # modelresults_path_ = dfs_path / public_variables.Modelresults_folder_
    # modelresults_combined_path_ = dfs_path / public_variables.Modelresults_combined_folder_
    # modelresults_combined_path_.mkdir(parents=True, exist_ok=True)
    # csv_filename = f'results_K{kfold}_{scoring[1]}_{RDKIT_descriptors}.csv'
    # append_modelResults_to_csv(ModelResults, modelresults_path_, csv_filename, concat_direction='hor')
    # csv_filename = f'results_K{kfold}_{scoring[1]}_{RDKIT_descriptors}_hor.csv'
    # ModelResults.to_csv(modelresults_combined_path_ / csv_filename, index=False)

    return

if __name__ == "__main__":
    
    main(public_variables.dfs_descriptors_only_path_)
    main(public_variables.dfs_reduced_path_)
    # main(public_variables.dfs_reduced_and_MD_path_)
    # main(public_variables.dfs_MD_only_path_)
