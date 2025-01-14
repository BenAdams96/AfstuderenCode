from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import random
from global_files import public_variables
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

def preprocess_dataframe(df):
    """Preprocess the dataframe by handling NaNs and standardizing."""
    # Handle NaNs: drop rows with NaNs or fill them
    df_cleaned = df.dropna()  # or df.fillna(df.mean())
    
    # Drop non-numeric target columns if necessary
    df_notargets = df_cleaned.drop(['mol_id', 'PKI', 'conformations (ns)'], axis=1, errors='ignore')
    
    # Standardize the dataframe
    scaler = StandardScaler()
    standardized_df = pd.DataFrame(scaler.fit_transform(df_notargets), columns=df_notargets.columns)
    
    return standardized_df

def calculate_correlation_matrix(df):
    """Calculate the correlation matrix of a standardized dataframe."""
    return df.corr()

def visualize_matrix(matrix, save_plot_path, idx, title_suffix=""):
    """Visualize a matrix (e.g., correlation matrix)."""
    print(save_plot_path)
    save_plot_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f'Matrix Visualization {title_suffix}')
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    plt.yticks(range(len(matrix.columns)), matrix.columns)
    plt.tight_layout()
    plt.savefig(save_plot_path / f'matrix_{idx}.png')
    # plt.show()

def compute_and_visualize_correlation_matrices_dic(dfs_path, exclude_files: list=None):
    """
    Compute and visualize the correlation matrices for a list of dataframes.
    
    Args:
        dfs (list): List of dataframes to process.
        save_plot_folder (Path): Folder path where to save the plots.
    
    Returns:
        list of tuples: Each tuple contains (original_df, df_notargets, st_df, correlation_matrix).
    """
    print(f'correlation matrix of {dfs_path}')
    dic = csv_to_dictionary.main(dfs_path,exclude_files=['concat_hor.csv','concat_ver.csv'])

    processed_dic = {}
    
    for name, df in dic.items():
        print(f'correlation matrix of: {name}')
        # Preprocess the dataframe: handle NaNs and standardize
        st_df = preprocess_dataframe(df)
        
        # Calculate and visualize correlation matrix for the standardized dataframe
        correlation_matrix = calculate_correlation_matrix(st_df)
        visualize_matrix(correlation_matrix, dfs_path, name, title_suffix="Original")
        
        # Store the results for potential further use
        processed_dic[name] = correlation_matrix
    return processed_dic

def main(dfs_path = public_variables.dfs_descriptors_only_path_):

    #just visualize the correlations matrices of the specified folder
    processed_dic = compute_and_visualize_correlation_matrices_dic(dfs_path)
    return

if __name__ == "__main__":
    main(dfs_path = public_variables.dfs_descriptors_only_path_)
    main(dfs_path = public_variables.dfs_reduced_path_)
    # main(dfs_path = public_variables.dfs_reduced_and_MD_path_)
