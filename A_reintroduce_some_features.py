from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import randomForest_read_in_models
import random
import public_variables
import csv_to_dataframes
import csv_to_dictionary
from csv_to_dataframes import csvfiles_to_dfs
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import itertools
from typing import List
import numpy as np
from pathlib import Path

import pandas as pd
import math
import re
import os

def standardize_dataframe(df):
    """Preprocess the dataframe by handling NaNs and standardizing."""
    # Handle NaNs: drop rows with NaNs or fill them
    df_cleaned = df.dropna()  # or df.fillna(df.mean())
    
    # Identify which non-feature columns to keep
    non_feature_columns = ['mol_id','PKI' 'conformations (ns)']
    existing_non_features = [col for col in non_feature_columns if col in df_cleaned.columns]
    
    # Drop non-numeric target columns if necessary
    features_df = df_cleaned.drop(columns=existing_non_features, axis=1, errors='ignore')
    
    # Standardize the dataframe
    scaler = StandardScaler()
    features_scaled_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)
    
    # Concatenate the non-feature columns back into the standardized dataframe
    standardized_df = pd.concat([df_cleaned[existing_non_features], features_scaled_df], axis=1)
    
    return standardized_df

def calculate_correlation_matrix(df):
    """Calculate the correlation matrix of a standardized dataframe."""
    df = df.drop(columns=['mol_id','PKI','conformations (ns)'], axis=1, errors='ignore')
    return df.corr()


def correlation_matrix_single_csv(df):
    # Preprocess the dataframe: handle NaNs and standardize
    st_df = standardize_dataframe(df)
    
    # Calculate and visualize correlation matrix for the standardized dataframe
    correlation_matrix = calculate_correlation_matrix(st_df)
    return st_df, correlation_matrix

def compute_correlation_matrices_of_dictionary(dfs_dictionary, exclude_files: list=None):
    """
    Compute and visualize the correlation matrices for a list of dataframes.
    
    Args:
        dfs (list): List of dataframes to process.
        save_plot_folder (Path): Folder path where to save the plots.
    
    Returns:
        list of tuples: Each tuple contains (original_df, df_notargets, st_df, correlation_matrix).
    """
    print(f'correlation matrix of {dfs_dictionary.keys()}')

    standardized_dfs_dic = {}
    correlation_matrices_dic = {}
    
    for name, df in dfs_dictionary.items():
        print(f'correlation matrix of: {name}')
        
        st_df, correlation_matrix = correlation_matrix_single_csv(df)
        
        # visualize_matrix(correlation_matrix, dfs_path, name, title_suffix="Original")
        
        # Store the results for potential further use
        standardized_dfs_dic[name] = st_df
        correlation_matrices_dic[name] = correlation_matrix
    return standardized_dfs_dic, correlation_matrices_dic

def identify_columns_to_drop(correlation_matrix, st_df, variances, threshold):
    """Identify columns to drop based on correlation threshold and variance."""
    corr_pairs = np.where(np.abs(correlation_matrix) > threshold)
    columns_to_drop = set()
    processed_pairs = set()

    for i, j in zip(*corr_pairs):
        if i != j:  # Skip self-correlation
            pair = tuple(sorted((i, j)))  # Ensure consistent ordering
            if pair not in processed_pairs:
                processed_pairs.add(pair)
                # Choose column to drop based on variance
                if variances[i] > variances[j]:
                    columns_to_drop.add(st_df.columns[j])
                else:
                    columns_to_drop.add(st_df.columns[i])
    
    return columns_to_drop

def identify_columns_to_drop_2_keep_lowest(correlation_matrix, df, variances, threshold):
    """Identify columns to drop based on correlation threshold and keeping the lowest indexed feature."""
    corr_pairs = np.where(np.abs(correlation_matrix) > threshold)
    columns_to_drop = set()
    processed_pairs = set()
    
    for i, j in zip(*corr_pairs):
        if i != j:  # Skip self-correlation
            pair = tuple(sorted((i, j)))  # Ensure consistent ordering
            if pair not in processed_pairs:
                processed_pairs.add(pair)
                # Drop the column with the higher index
                if i < j:
                    columns_to_drop.add(df.columns[j])  # Drop column j (higher index)
                else:
                    columns_to_drop.add(df.columns[i])  # Drop column i (higher index)
    
    return columns_to_drop

def get_reduced_features_for_dataframes_in_dic(correlation_matrices_dic, dfs_dictionary, threshold):
    """
    Reduce dataframes based on correlation and visualize the reduced matrices.
    
    Args:
        correlation_matrices_dic (dict): Dictionary of correlation matrices.
        dfs_dictionary (dict): Dictionary of dataframes corresponding to the correlation matrices.
        threshold (float): Correlation threshold for dropping features.
        save_plot_folder_reduced (Path): Folder path where to save the reduced matrix plots.
    
    Returns:
        dict: Dictionary of reduced dataframes.
    """
    reduced_dfs_dictionary = {}
    
    for key in correlation_matrices_dic.keys():
        # Calculate variances for the non-standardized dataframe
        
        
        # Identify non-feature columns to retain
        non_feature_columns = ['mol_id','PKI','conformations (ns)']
        existing_non_features = [col for col in non_feature_columns if col in dfs_dictionary[key].columns]
        
        # Drop only the features for correlation analysis
        features_df = dfs_dictionary[key].drop(columns=existing_non_features, axis=1)
        variances = features_df.var()

        # Identify columns to drop based on high correlation and variance
        columns_to_drop = identify_columns_to_drop_2_keep_lowest(correlation_matrices_dic[key], features_df, variances, threshold)
        
        # Create the reduced dataframe by including the retained non-feature columns
        reduced_df = pd.concat([dfs_dictionary[key][existing_non_features], features_df], axis=1)
        reduced_df = reduced_df.drop(columns=columns_to_drop, axis=1)

        reduced_dfs_dictionary[key] = reduced_df
    
    return reduced_dfs_dictionary

def save_dataframes_to_csv(dic_with_dfs,save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in dic_with_dfs.items():
        print(f"save dataframe: {name}")
        df.to_csv(save_path / f'{name}.csv', index=False)

def save_reduced_dataframes(dfs, base_path):
    dir = public_variables.dfs_reduced_path_
    final_path = base_path / dir
    final_path.mkdir(parents=True, exist_ok=True)
    timeinterval = public_variables.timeinterval_snapshots

    for i, x in enumerate(np.arange(0, len(dfs) * timeinterval, timeinterval)):
        if x.is_integer():
            x = int(x)
        print(f"x: {x}, i: {i}")
        dfs[i].to_csv(final_path / f'{x}ns.csv', index=False)

def load_results(csv_path):
    df = pd.DataFrame()
    return df

def remove_constant_columns_from_dfs(dfs_dictionary):
    cleaned_dfs = {}
    
    for key, df in dfs_dictionary.items():
        # Identify constant columns
        constant_columns = df.columns[df.nunique() <= 1]
        
        if not constant_columns.empty:
            print(f"In '{key}', the following constant columns were removed: {', '.join(constant_columns)}")
        # Remove constant columns and keep only non-constant columns
        non_constant_columns = df.loc[:, df.nunique() > 1]
        cleaned_dfs[key] = non_constant_columns
    return cleaned_dfs

def reintroduce_top_correlated_features(top_features, correlation_matrix, original_df, correlation_threshold=0.8, num_top_features=5):
    features_to_reintroduce = set()
    print(top_features)
    print(correlation_matrix)
    original_df = original_df.drop(['mol_id', 'PKI', 'conformations (ns)'], axis=1, errors='ignore')
    # Get the top N features based on importance
    top_feature_indices = top_features.index[:num_top_features]
    
    for feature in top_feature_indices:
        correlations = correlation_matrix[feature]
        
        # Find strongly correlated features, excluding the feature itself
        correlated_features = correlations[correlations.abs() >= correlation_threshold].index
        correlated_features = correlated_features[correlated_features != feature]
        
        print(f'Analyzing Feature: {feature}')
        print(f'Correlated Features: {correlated_features.tolist()}')
        
        # Collect correlated values
        correlated_values = correlations[correlated_features]
        
        # Keep track of the lowest correlated feature that is not in top_features
        for _ in range(len(correlated_features)):
            # Get the feature with the lowest absolute correlation
            lowest_corr_feature = correlated_values.abs().idxmin()
            lowest_corr_value = correlated_values[lowest_corr_feature]
            
            # Check if it's already in the reduced features
            if lowest_corr_feature not in top_features.index:
                print(f'Adding feature: {lowest_corr_feature} with correlation value: {lowest_corr_value}')
                features_to_reintroduce.add(lowest_corr_feature)
                break  # Exit the loop once we found a valid feature

            # Remove the feature from the correlated_values to check the next one
            correlated_values = correlated_values.drop(lowest_corr_feature)

    # Filter the original DataFrame to include only the reintroduced features
    reintroduced_df = original_df.loc[:, original_df.columns.isin(features_to_reintroduce)]

    print(reintroduced_df)
    return reintroduced_df

def main(dfs_reduced_path = public_variables.dfs_reduced_path_):
    
    # dfs_dictionary = csv_to_dictionary.main(dfs_reduced_path,exclude_files=['concat_hor.csv','concat_ver.csv', 'big.csv'])#,'conformations_1000.csv','conformations_1000_molid.csv'])
    # print(dfs_dictionary.keys())
    # dfs_dictionary = remove_constant_columns_from_dfs(dfs_dictionary)
    
    
    # standardized_dfs_dic, correlation_matrices_dic = compute_correlation_matrices_of_dictionary(dfs_dictionary)
    
    # # Reduce the dataframes based on correlation
    # reduced_dfs_in_dic = get_reduced_features_for_dataframes_in_dic(correlation_matrices_dic, dfs_dictionary, threshold=correlation_threshold)
    # #reduced dataframes including mol_ID and PKI. so for 0ns 1ns etc. 
    # save_dataframes_to_csv(reduced_dfs_in_dic, save_path=dfs_reduced_path)

    models_dict = randomForest_read_in_models.read_in_model_dictionary(dfs_reduced_path / f'{public_variables.Modelresults_folder_}RF','original_models_dic.pkl')
    dfs_dictionary = csv_to_dictionary.main(public_variables.dfs_descriptors_only_path_,exclude_files=['concat_hor.csv','concat_ver.csv', 'big.csv','conformations_1000.csv','conformations_1000_molid.csv'])
    for key, model in models_dict['RF_Allmodels_k10_R-squared (R²)']['original models'].items():
        print(f'{key = }')
        top_2_features_of_model = model.top_features.head(2)
        
        df = dfs_dictionary[key]
        print(df.columns)
        st_df, correlation_matrix = correlation_matrix_single_csv(df)
        reintroduced_df = reintroduce_top_correlated_features(top_features = model.top_features, correlation_matrix=correlation_matrix, original_df=df)
        print(reintroduced_df)
        print('done')
    return

if __name__ == "__main__":
    main(dfs_reduced_path = public_variables.dfs_reduced_path_)
    
    # bigdf = pd.read_csv(public_variables.dfs_descriptors_only_path_ / 'total_df_ordered_by_conformation.csv')
    # dic = {}
    # dic['total_df_ordered_by_conformation.csv'] = bigdf
    # standardized_dfs_dic, correlation_matrices_dic = compute_correlation_matrices_of_dictionary(dic)
    # print(correlation_matrices_dic)
    # reduced_dfs_in_dic = get_reduced_features_for_dataframes_in_dic(correlation_matrices_dic, dic, threshold=0.85)
    # save_dataframes_to_csv(reduced_dfs_in_dic, save_path=public_variables.dfs_reduced_path_)

