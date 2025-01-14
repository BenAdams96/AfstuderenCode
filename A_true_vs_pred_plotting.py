from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import randomForest_read_in_models

from global_files import public_variables, csv_to_dictionary

from sklearn.preprocessing import StandardScaler
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

def plot_predicted_vs_real_pKi(y_true, y_pred,name, save_path):
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
    plt.savefig(save_path / f'{name}.png')

def plot_avg_predicted_vs_real_pKi(df, name, save_path):
    """
    Plots the average predicted pKi values against real pKi values for model evaluation.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns 'mol_id', 'True_pKi', and 'Predicted_pKi'.
        name (str): The name for the plot file.
        save_path (Path): Path to save the plot.
    """
    # Aggregate by mol_id to compute the average Predicted_pKi and get the True_pKi
    avg_df = df.groupby('mol_id', as_index=False).agg({
        'True_pKi': 'first',  # Assuming True_pKi is the same for all conformations of the same mol_id
        'Predicted_pKi': 'mean'
    })
    
    # Extract true and predicted values
    y_true = avg_df['True_pKi']
    y_pred = avg_df['Predicted_pKi']
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='k', s=80)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')  # Ideal line
    plt.xlabel('Real pKi Values')
    plt.ylabel('Average Predicted pKi Values')
    plt.title('Average Predicted vs Real pKi Values')
    plt.grid(True)
    
    # Save the plot
    save_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    plt.savefig(save_path / f'{name}_average.png')

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

def main(final_path):
    plot_path = final_path / 'plots'
    plot_path.mkdir(parents=True, exist_ok=True)

    dfs_in_dic_t_vs_p = csv_to_dictionary.csvfiles_to_dic(final_path)
    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic_t_vs_p.keys())) #RDKIT first
    dfs_in_dic_t_vs_p = {key: dfs_in_dic_t_vs_p[key] for key in sorted_keys_list if key in dfs_in_dic_t_vs_p} #order
    print(dfs_in_dic_t_vs_p.keys())

    for name, df in dfs_in_dic_t_vs_p.items():
        true_pKi = df['True_pKi']
        predicted_pKi = df['Predicted_pKi']
        if 'conformations' in name:
            # Call the function for average predicted vs real pKi
            plot_avg_predicted_vs_real_pKi(df, name, plot_path)
            plot_predicted_vs_real_pKi(true_pKi, predicted_pKi, name, plot_path)
        else:
            # Call the original function for predicted vs real pKi
            plot_predicted_vs_real_pKi(true_pKi, predicted_pKi, name, plot_path)
    return

if __name__ == "__main__":
    base_path_ = Path(__file__).resolve().parent.parent
    model_ = 'RF'
    RDKIT_descriptors_ = 'WHIM' #choose between 'WHIM' and 'GETAWAY' #VARIABLE
    dataset_protein_ = 'JAK1'   #VARIABLE 'JAK1' or 'GSK3'

    dataframes_master_ = base_path_ / Path(f'dataframes_{dataset_protein_}_{RDKIT_descriptors_}')
    subgroup_path = dataframes_master_ / 'descriptors only'
    Modelresults_folder_ = f'ModelResults_{model_}'

    final_path = subgroup_path / Modelresults_folder_ / 'true_vs_prediction'
    main(final_path)