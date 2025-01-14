import randomForest_read_in_models

from sklearn.preprocessing import StandardScaler

from global_files import public_variables,csv_to_dictionary
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
import numpy as np
from pathlib import Path
import shutil
from itertools import product

import pandas as pd
import math
import re
import os

def extract_k_and_scoring(filename):
    # Use regex to find the pattern 'k' followed by digits and the scoring method
    match = re.search(r'k(\d+)_(.+)\.pkl', filename)
    if match:
        k_value = int(match.group(1))
        scoring_metric = match.group(2)
        return k_value, scoring_metric
    else:
        return None, None
    
def get_molecules_lists_temp(parent_path):
    folder = public_variables.dfs_descriptors_only_path_
    csv_file = '0ns.csv'
    final_path = parent_path / folder / csv_file
    molecules_list = []
    invalid_mols = []
    df = pd.read_csv(final_path)
    mol_id_column = df['mol_id']

    valid_mol_list_str = list(map(str, mol_id_column))
    print(valid_mol_list_str)
    print(len(valid_mol_list_str))
    return  valid_mol_list_str

def extract_number(filename):
    return int(filename.split('ns.csv')[0])

def copy_redfolder_only_csv_files(source_folder, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            # Copy each CSV file to the destination folder
            shutil.copy2(source_file, destination_file)

def concatenate_group(group):
    # Separate out the PKI and mol_id
    mol_id = group['mol_id'].iloc[0]
    pki_value = group['PKI'].iloc[0]
    
    # Drop 'PKI' and 'mol_id' from the group to avoid duplication in concatenation
    group = group.drop(columns=['mol_id', 'PKI'])
    
    # Concatenate the remaining columns horizontally (axis=1)
    concatenated = pd.concat([group.reset_index(drop=True)], axis=1).T.reset_index(drop=True)
    
    # Flatten the column names
    concatenated.columns = [f"{col}_{i}" for i, col in enumerate(concatenated.columns)]
    
    # Add the 'mol_id' and 'PKI' back to the start
    concatenated.insert(0, 'mol_id', mol_id)
    concatenated.insert(1, 'PKI', pki_value)
    
    return concatenated

def create_dataframes_MD_only():
    reduced_MD_dataframes_folder = public_variables.dfs_reduced_and_MD_path_
    MD_only_folder = public_variables.dfs_MD_only_path_

    if MD_only_folder.exists():
        if MD_only_folder.is_dir():
            print('already MD folder')
            # Remove the existing destination folder
            shutil.rmtree(MD_only_folder)

    os.makedirs(MD_only_folder, exist_ok=True)

    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic(reduced_MD_dataframes_folder, exclude_files=['concat_ver.csv', 'concat_hor.csv','rdkit_min.csv','MD_output.csv', 'conformations_1000.csv']) # , '0ns.csv', '1ns.csv', '2ns.csv', '3ns.csv', '4ns.csv', '5ns.csv', '6ns.csv', '7ns.csv', '8ns.csv', '9ns.csv', '10ns.csv'
    dfs_in_dic_concat = csv_to_dictionary.csvfiles_to_dic(reduced_MD_dataframes_folder, exclude_files=['rdkit_min', '0ns', '1ns', '2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '8ns', '9ns', '10ns'])

    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys()))
    dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic}

    #csv_files = sorted([f for f in os.listdir(reduced_dataframes_folder) if f.endswith('.csv')], key=extract_number)

    for name, df in list(dfs_in_dic.items()):
        if name.startswith('conformations'):
            df_cleaned = df.loc[:, ~df.columns.str.isnumeric()]
            df_cleaned.to_csv(MD_only_folder / Path(name + '.csv'), index=False)
            print(f'done with {name}')
        elif name.endswith('ns'):
            df_cleaned = df.loc[:, ~df.columns.str.isnumeric()]
            df_cleaned.to_csv(MD_only_folder / Path(name + '.csv'), index=False)
            print(f'done with {name}')
        else:
            df.to_csv(MD_only_folder / Path(name + '.csv'), index=False)
            continue
    return



def main():
    #Works by copying the reduced+MD csv files and removes all features that are numeric.
    create_dataframes_MD_only()

    return

if __name__ == "__main__":
    main()
