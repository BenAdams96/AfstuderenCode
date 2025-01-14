# Standard library imports
import os
import re

# Third-party libraries
import numpy as np
import pandas as pd
from pathlib import Path

# RDKit imports
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Project-specific imports
from global_files import public_variables, public_functions
import trj_to_pdbfiles

def create_full_dfs(folder_path_clustered_pdbs, molID_PKI_df, descriptors):
    ''' Create the WHIM dataframes for every molecule for every timestep (xdirs)
        First goes over the timesteps, then every molecule within that timestep
        Output: total_df_conf_order - dataframe with the descriptors of the molecule for every timestep including mol_id and PKI
    '''
    if descriptors == 'WHIM':
        num_columns = 114
    elif descriptors == 'GETAWAY':
        num_columns = 273
    else:
        raise ValueError("Error: Choose a valid descriptor")
    
    if public_variables.dataset_protein_ == 'JAK1':
        max_molecule_number = 615
    elif public_variables.dataset_protein_ == 'GSK3':
        max_molecule_number = 856

    rows = []


    # List all PDB files in the directory
    pdb_files = [file for file in os.listdir(folder_path_clustered_pdbs) if file.endswith('.pdb')]
    filtered_sorted_list = sorted([file for file in pdb_files if int(file.split('_')[0]) <= max_molecule_number], #TODO: shitty solution
                        key=lambda x: int(x.split('_')[0]))
    
    # print(filtered_sorted_list)
    for pdb_file in filtered_sorted_list:
        #print(pdb_file) #is 001_centroid.pdb
        dir_path = folder_path_clustered_pdbs
        pdb_file_path = os.path.join(dir_path, pdb_file)
        print(pdb_file)
        #print(pdb_file_path) #is 001_centroid.pdb
        mol = Chem.MolFromPDBFile(pdb_file_path, removeHs=False, sanitize=False)
        
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except ValueError as e:
                print(f"Sanitization error: {e}")
                print(pdb_file)
        else:
            print("Invalid molecule:")
            print(pdb_file)
            continue
        
        # Calculate descriptors
        if descriptors == 'WHIM':
            mol_descriptors = rdMolDescriptors.CalcWHIM(mol)
        elif descriptors == 'GETAWAY':
            mol_descriptors = rdMolDescriptors.CalcGETAWAY(mol)
        
        # index_to_insert = int(pdb_file[:3]) + int((float(dir_path.name.rstrip('ns')) / 10) * (len(sorted_folders) - 1) * len(all_molecules_list))
        pki_value = molID_PKI_df.loc[molID_PKI_df['mol_id'] == int(pdb_file[:3]), 'PKI'].values[0]
        centroid_timepoint = pdb_file.split('_')[1].split('.')[0]
        conformation_value = int(centroid_timepoint)/1000
        # Collect the row data
        rows.append([int(pdb_file[:3]), pki_value, conformation_value] + mol_descriptors)

    # Convert rows list to DataFrame
    columns = ['mol_id', 'PKI', 'conformations (ns)'] + list(range(num_columns))
    total_df_conf_order = pd.DataFrame(rows, columns=columns).dropna().reset_index(drop=True)
    return total_df_conf_order

def get_sorted_folders(base_path):
    '''The folder with CSV files 'dataframes_JAK1_WHIM' is the input and these CSV files will be
       put into a list of pandas DataFrames, also works with 0.5 1 1.5 formats.
    '''
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    sorted_folders = []
    # Define the regex pattern to match filenames like '0ns.csv', '1ns.csv', '1.5ns.csv', etc.
    pattern = re.compile(r'^\d+(\.\d+)?ns$')

    for csv_file in sorted(base_path.glob('*'), key=lambda x: extract_number(x.name)):
        if pattern.match(csv_file.name):  # Check if the file name matches the pattern
            sorted_folders.append(csv_file)
        else:
            sorted_folders.insert(0,csv_file)
    return sorted_folders

def extract_number(filename):
    # Use regular expression to extract numeric part (integer or float) before 'ns.csv'
    match = re.search(r'(\d+(\.\d+)?)ns$', filename)
    if match:
        number_str = match.group(1)
        # Convert to float first
        number = float(number_str)
        # If it's an integer, convert to int
        if number.is_integer():
            return int(number)
        return number
    else:
        return float('inf')

def remove_invalids_from_dfs(dic_with_dfs,invalids):
    invalids = list(map(int, invalids))
    print(invalids)
    filtered_dic_with_dfs = {}
    for name, df in dic_with_dfs.items():
        filtered_df = df[~df['mol_id'].isin(invalids)]
        filtered_dic_with_dfs[name] = filtered_df
    return filtered_dic_with_dfs

def save_dataframes(dic_with_dfs,base_path):
    dir = public_variables.dfs_descriptors_only_path_
    final_path = base_path / dir
    final_path.mkdir(parents=True, exist_ok=True)
    timeinterval = public_variables.timeinterval_snapshots
    #TODO: use a dictionary.
    for name, df in dic_with_dfs.items():
        #print(f"name: {name}, i: {df.head(1)}")
        df.to_csv(final_path / f'{name}.csv', index=False)

def get_filtered_and_sorted_folders(base_folder, target_filter=None, cluster_filter=None):
    # Get all folders in the base directory
    all_folders = [
        f for f in os.listdir(base_folder) 
        if os.path.isdir(os.path.join(base_folder, f))
    ]

    # Regex to extract target and cluster values
    folder_pattern = re.compile(r'clustering_target(\d+(?:\.\d+)?)%_cluster(\d+)')

    # Extract and parse target and cluster values
    folder_info = []
    for folder in all_folders:
        match = folder_pattern.search(folder)
        if match:
            target = float(match.group(1))
            cluster = int(match.group(2))
            folder_info.append((folder, target, cluster))

    # Filter based on target_filter and cluster_filter
    if target_filter is not None:
        folder_info = [
            (folder, target, cluster) for folder, target, cluster in folder_info 
            if target in target_filter
        ]
    if cluster_filter is not None:
        folder_info = [
            (folder, target, cluster) for folder, target, cluster in folder_info 
            if cluster in cluster_filter
        ]

    # Sort folders: target descending, cluster ascending
    folder_info.sort(key=lambda x: (-x[1], x[2]))

    # Return full paths of sorted folders
    return [os.path.join(base_folder, folder) for folder, _, _ in folder_info]

#NOTE: this file does: get targets, count how many valid molecules and which,it creates the folder 'dataframes_WHIMJAK1' or equivellant
def main(base_path,smiles_activity_dataset, RDKIT_descriptors):
    df_targets = public_functions.get_all_targets(smiles_activity_dataset) #df with columns: 'mol_id' and 'PKI value'. all molecules
    clustering_folder = public_variables.base_path_ / 'clustering folder' / public_variables.dataset_protein_
    clustering_folder.mkdir(parents=True, exist_ok=True)
    target_filter = [50,40,30,20,10]  # Only include targets 10 and 20
    cluster_filter = [1,2,3,4,5,6,7,8,9,10]

    #add like a filter which ones i want. yes
    sorted_folders = get_filtered_and_sorted_folders(clustering_folder, target_filter, cluster_filter)
    print(sorted_folders)
    for folder in sorted_folders:
        #create the dataframes, which eventually will be placed in 'dataframes_JAK1_WHIM' and also add the targets to the dataframes.
        df = create_full_dfs(folder, df_targets, RDKIT_descriptors)
        public_variables.dfs_descriptors_only_path_.mkdir(parents=True, exist_ok=True)
        df.to_csv(public_variables.dfs_descriptors_only_path_ / f'{Path(folder).name}.csv', index=False)
    # df_sorted_by_molid.to_csv(public_variables.dataframes_master_ / 'initial_dataframe_mol_id.csv', index=False)
    return

if __name__ == "__main__":
    base_path = public_variables.base_path_
    dataset_csvfile_path = public_variables.dataset_path_ # 'JAK1dataset.csv'
    RDKIT_descriptors = public_variables.Descriptor_

    main(base_path, dataset_csvfile_path, RDKIT_descriptors)

