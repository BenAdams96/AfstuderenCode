import math
import numpy as np
import os
import MDAnalysis as mda
from MDAnalysis.coordinates import PDB
import rdkit
import public_variables
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdMolDescriptors
import trj_to_pdbfiles
import pandas as pd
from pathlib import Path
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import re
import pathlib

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

def calculate_molecular_weight_from_pdb(pdb_file):
    """
    Calculate the molecular weight of a molecule from a PDB file.
    
    Parameters:
    - pdb_file: str, path to the PDB file
    
    Returns:
    - molecular_weight: float, molecular weight of the molecule
    """
    # Read the molecule from the PDB file
    mol = Chem.MolFromPDBFile(pdb_file)
    if mol is not None:
        return Descriptors.ExactMolWt(mol)  # Use the correct import
    else:
        print(f"Could not read molecule from {pdb_file}.")
        return None

def calculate_molecular_weights(folder_path):
    """
    Calculate the molecular weights of all PDB files in a specified folder.
    
    Parameters:
    - folder_path: str, path to the folder containing PDB files
    
    Returns:
    - weights: list of tuples, each containing (filename_without_extension, molecular_weight)
    """
    weights = []
    for filename in os.listdir(folder_path):
        
        if filename.endswith('.pdb'):
            pdb_file_path = os.path.join(folder_path, filename)
            mol_weight = calculate_molecular_weight_from_pdb(pdb_file_path)
            if mol_weight is not None:
                # Create a tuple with the filename without the extension and the molecular weight as an integer
                weights.append((int(filename[:-4]), mol_weight))  # Remove the last 4 characters (.pdb)

    return weights

def save_dataframes(dic_with_dfs, save_path = public_variables.dfs_descriptors_only_path_):
    save_path.mkdir(parents=True, exist_ok=True)
    timeinterval = public_variables.timeinterval_snapshots
    
    for name, df in dic_with_dfs.items():
        #print(f"name: {name}, i: {df.head(1)}")
        df.to_csv(save_path / f'{name}.csv', index=False)


def create_dfs_dic(totaldf, timeinterval = public_variables.timeinterval_snapshots):

    df_dict = {}

    # Loop over the range from range_start to range_end (inclusive)
    for i in np.arange(0,10+timeinterval,timeinterval):
        i = round(i, 2)
        if i.is_integer():
            i = int(i)
        # Create a new dataframe with rows where 'conformations (ns)' == i
        filtered_df = totaldf[totaldf['conformations (ns)'] == i].copy()

        # Drop the 'conformations (ns)' column
        filtered_df.drop(columns=['conformations (ns)'], inplace=True)
        
        # # Store the dataframe in the dictionary with a key like '0ns', '1ns', etc.
        df_dict[f'{i}ns'] = filtered_df.reset_index(drop=True)

    return df_dict

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

def main():
    descriptors_only_scaled_path = public_variables.dataframes_master_ / 'descriptors only scaled mw'
    descriptors_only_scaled_path.mkdir(parents=True, exist_ok=True)

    folder_path = public_variables.base_path_ / 'pdb'
    molecular_weights = calculate_molecular_weights(folder_path)
    molecular_weights_df = pd.DataFrame(molecular_weights, columns=['mol_id', 'molecular_weight'])
    
    # bigdf = pd.read_csv(public_variables.dataframes_master_ / 'conformations_1000.csv')
    big_df = pd.read_csv(public_variables.dfs_descriptors_only_path_ / 'conformations_1000.csv')
    merged_dataframe = pd.merge(big_df, molecular_weights_df, on='mol_id', how='left')
    merged_dataframe.to_csv(descriptors_only_scaled_path / 'conformations_1000.csv', index=False)

    st_df, correlation_matrix = correlation_matrix_single_csv(merged_dataframe)
    
    correlation_with_mw = correlation_matrix['molecular_weight']

    # Identify features with correlation >= 0.8 (excluding 'molecular_weight' itself)
    correlated_features = correlation_with_mw[correlation_with_mw.abs() >= 0.8].index.tolist()
    correlated_features.remove('molecular_weight')  # Remove the molecular_weight itself if it's included
    
    # Divide those columns by molecular_weight
    for feature in correlated_features:
        merged_dataframe[feature] = merged_dataframe[feature] / merged_dataframe['molecular_weight']
    merged_dataframe_without_mw = merged_dataframe.drop(columns=['molecular_weight'])

    # Save the updated DataFrame to a CSV file
    merged_dataframe_without_mw.to_csv(descriptors_only_scaled_path / 'conformations_1000.csv', index=False)


    dfs_in_dict = create_dfs_dic(merged_dataframe_without_mw, timeinterval = 1)
    save_dataframes(dfs_in_dict,descriptors_only_scaled_path)
    
    if not descriptors_only_scaled_path.exists():
        print(f"Error: The path '{descriptors_only_scaled_path}' does not exist.")
        return
    
    timeinterval = [1,0.5,0.2,0.1,0.05,0.02]

    for t in timeinterval:
        print(t)
        reduced_dataframe = reduce_conformations(merged_dataframe_without_mw, interval=t)
        reduced_dataframe.to_csv(descriptors_only_scaled_path / f'conformations_{int(10/t)}.csv', index=False)
        print('done')

    return

if __name__ == "__main__":

    main()


# print("hello world")
# print(rdkit.__version__)
# pdb_file = '100.pdb'
# mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)

# if mol is None:
#     print("Failed to read PDB file")
# else:
#     whim_descriptors = rdMolDescriptors.CalcWHIM(mol)
#     print(len(whim_descriptors))
#     #for i, value in enumerate(whim_descriptors):
#     #    print(f"WHIM Descriptor {i+1}: {value}")

