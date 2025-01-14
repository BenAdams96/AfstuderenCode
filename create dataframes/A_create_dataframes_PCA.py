from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import random
from global_files import public_variables, csv_to_dictionary
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
import itertools
from typing import List
import numpy as np
from pathlib import Path
from typing import Dict

import pandas as pd
import math
import re
import os

def standardize_dataframes(df_dict):
    """
    Standardizes all numeric columns in the DataFrames of a dictionary, excluding specified columns.
    
    Parameters:
        df_dict (dict): A dictionary where keys are strings and values are pandas DataFrames.
        exclude_columns (list): List of column names to exclude from standardization.
    
    Returns:
        dict: A dictionary with the same keys, but with numeric columns standardized and excluded columns retained.
    """
    standardized_dict = {}
    exclude_columns=['mol_id', 'PKI', 'conformations (ns)']
    
    for key, df in df_dict.items():
        # Identify numeric columns excluding the specified columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_columns)
        
        # Apply standardization to numeric columns only
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[numeric_cols])
        
        # Create a DataFrame with standardized numeric columns
        standardized_numeric_df = pd.DataFrame(
            scaled_values,
            columns=numeric_cols,
            index=df.index
        )
        
        # Combine standardized numeric columns with excluded columns
        valid_exclude_columns = [col for col in exclude_columns if col in df.columns]
        
        # Concatenate the excluded columns (mol_id, PKI, etc.) with the standardized numeric columns
        standardized_df = pd.concat([df[valid_exclude_columns], standardized_numeric_df[numeric_cols]], axis=1)
        
        # Ensure column order matches the original DataFrame
        standardized_df = standardized_df[df.columns]
        
        # Add the standardized DataFrame to the dictionary
        standardized_dict[key] = standardized_df
    
    return standardized_dict

def run_pca_full(standardized_dfs):
    """
    Runs PCA for all components on each DataFrame and collects cumulative variance.
    
    Args:
        standardized_dfs: Dictionary of standardized DataFrames.
    
    Returns:
        A dictionary containing PCA models and cumulative explained variance for each DataFrame.
    """
    pca_results = {}
    exclude_columns=['mol_id', 'PKI', 'conformations (ns)']
    for key, standardized_df in standardized_dfs.items():
        pca_input = standardized_df.drop(columns=exclude_columns, axis=1, errors='ignore')
        pca = PCA()  # Run PCA without limiting n_components
        pca.fit(pca_input)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        pca_results[key] = pca
        
    
    return pca_results

def plot_elbow_all(pca_results: Dict[str, PCA], max_components=None):
    """
    Plots the cumulative explained variance for each dataset in the PCA results.

    Args:
        pca_results: Dictionary containing PCA models for each DataFrame.
        max_components: Maximum number of components to display (default: None, shows all).
    """
    for key, pca in pca_results.items():
        # Extract cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        total_components = len(cumulative_variance)
        
        # Set limit for the number of components to plot
        if max_components is None or max_components > total_components:
            max_components = total_components
        
        # Plot the cumulative variance
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(1, max_components + 1), 
            cumulative_variance[:max_components], 
            marker='o', label='Cumulative Variance'
        )
        
        # Add plot details
        plt.title(f'Cumulative Explained Variance - {key}')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.axhline(0.97, color='red', linestyle='--', label='95% Variance Threshold')
        plt.legend()
        plt.savefig(public_variables.dataframes_master_ / 'PCA' / f'{key}.png')


def apply_pca_to_standardized_dfs(standardized_dfs, n_components=0.97):
    pca_results = {}
    exclude_columns = ['mol_id', 'PKI', 'conformations (ns)']

    for key, standardized_df in standardized_dfs.items():
        # Apply PCA to each standardized DataFrame
        
        # Exclude the specified columns (mol_id, PKI, conformations (ns))
        pca_input = standardized_df.drop(columns=exclude_columns, axis=1, errors='ignore')

        # Initialize PCA
        pca = PCA()
        pca.fit(pca_input)
        
        # Cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Determine the number of components to retain based on the threshold
        if isinstance(n_components, float):
            # Case 1: n_components is a float (e.g., 0.95 means 95% variance)
            if 0 < n_components < 1:
                n_components_ = np.argmax(cumulative_variance >= n_components) + 1
            else:
                raise ValueError("n_components as float should be between 0 and 1.")
        
        elif isinstance(n_components, int):
            # Case 2: n_components is an integer (e.g., 20 means top 20 components)
            n_components_ = n_components
            if n_components_ > pca_input.shape[1]:
                raise ValueError(f"n_components ({n_components_}) cannot be greater than the number of features ({pca_input.shape[1]}).")
        
        # Apply PCA with the selected number of components
        pca_final = PCA(n_components=n_components_)
        data_pca = pca_final.fit_transform(pca_input)
        
        # Create a DataFrame for the PCA results
        data_pca_df = pd.DataFrame(
            data_pca,
            columns=[f"PC{i+1}" for i in range(data_pca.shape[1])],  # Name columns as PC1, PC2, etc.
            index=standardized_df.index  # Use the original DataFrame's index
        )

        # Select only the valid exclude columns
        valid_exclude_columns = [col for col in exclude_columns if col in standardized_df.columns]

        # Concatenate the original exclude columns and the PCA results
        data_pca_df_complete = pd.concat([standardized_df[valid_exclude_columns], data_pca_df], axis=1)
        
        # Store the result in the pca_results dictionary
        pca_results[key] = data_pca_df_complete
    
    return pca_results


def calculate_correlation_matrix(df):
    """Calculate the correlation matrix of a standardized dataframe."""
    df = df.drop(columns=['mol_id','PKI','conformations (ns)'], axis=1, errors='ignore')
    return df.corr()


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

def main(dfs_path = public_variables.dfs_descriptors_only_path_):
    dfs_PCA_path = public_variables.dataframes_master_ / f'PCA'  # e.g., 'dataframes_JAK1_WHIM
    dfs_PCA_path.mkdir(parents=True, exist_ok=True)

    dfs_dictionary = csv_to_dictionary.main(dfs_path,exclude_files=['concat_hor.csv','concat_ver.csv', 'big.csv','conformations_200.csv','conformations_500.csv','conformations_1000.csv','conformations_1000_molid.csv'])#,'conformations_1000.csv','conformations_1000_molid.csv'])

    print(dfs_dictionary.keys())
    dfs_dictionary = remove_constant_columns_from_dfs(dfs_dictionary)

    dfs_standardized = standardize_dataframes(dfs_dictionary)
    
    dfs_pca = run_pca_full(dfs_standardized)
    plot_elbow_all(dfs_pca, max_components=None)
    pca_results = apply_pca_to_standardized_dfs(dfs_standardized, n_components=0.97)
    print(pca_results)
    save_dataframes_to_csv(pca_results, dfs_PCA_path)

    
    
    
    
    # # Reduce the dataframes based on correlation
    # reduced_dfs_in_dic = get_reduced_features_for_dataframes_in_dic(correlation_matrices_dic, dfs_dictionary, threshold=correlation_threshold)
    # #reduced dataframes including mol_ID and PKI. so for 0ns 1ns etc. 
    # save_dataframes_to_csv(reduced_dfs_in_dic, save_path=dfs_PCA)
    return

if __name__ == "__main__":
    main(dfs_path = public_variables.dfs_descriptors_only_path_)
    
    # bigdf = pd.read_csv(public_variables.initial_dataframe)
    # dic = {}
    # dic['conformations_1000.csv'] = bigdf
    # standardized_dfs_dic, correlation_matrices_dic = compute_correlation_matrices_of_dictionary(dic)
    # print(correlation_matrices_dic)
    # reduced_dfs_in_dic = get_reduced_features_for_dataframes_in_dic(correlation_matrices_dic, dic, threshold=0.65)
    # save_dataframes_to_csv(reduced_dfs_in_dic, save_path=public_variables.dfs_reduced_path_)











