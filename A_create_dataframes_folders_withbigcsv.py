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
import re
import pathlib

def get_targets(dataset):
    """read out the original dataset csv file and get the targets + convert exp_mean to PKI
    out: pandas dataframe with two columns: ['mol_id','PKI']
    """
    df = pd.read_csv(dataset)
    df['PKI'] = -np.log10(df['exp_mean [nM]'] * 1e-9)
    return df[['mol_id','PKI']]

def count_ns_dirs(ligandconformations_path):
    '''count how many directories there are in ligand_conformations_for_every_ns (so how many time steps)
        in: ligandconformations_path
        out: how many directories there are (for example 11 nanosecond snapshots)
    '''
    try:
        # List all entries in the directory
        entries = os.listdir(ligandconformations_path)
        # Count only directories
        count = sum(1 for entry in entries if os.path.isdir(os.path.join(ligandconformations_path, entry)))
        return count
    except FileNotFoundError:
        print(f"Error: The directory '{ligandconformations_path}' does not exist.")
        return 0
    except PermissionError:
        print(f"Error: Permission denied to access '{ligandconformations_path}'.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0
    
def create_full_dfs(ligand_conformations_path, molID_PKI_df, descriptors, all_molecules_list):
    ''' create the WHIM dataframes for every molecule for every timestep (xdirs)
        first goes over the timesteps, then every molecule within that timestep
        output: dfs: x amount of dataframes with the descriptors of the molecule for every timestep including mol_id and PKI
    '''
    dfs = []
    dic_with_dfs = {}
    if descriptors == 'WHIM':
        num_columns = 114
    elif descriptors == 'GETAWAY':
        num_columns = 273
    else:
        raise ValueError("Error: Choose a valid descriptor")
    
    #NOTE: RDKIT will be placed in the first index
    sorted_folders = get_sorted_folders(ligand_conformations_path) #all folders/paths in 'ligand_conformations_JAK1' sorted from 0ns to 10ns
    print(len(sorted_folders))
    total_df_conf_order = pd.DataFrame(index=range(0, len(sorted_folders) * len(all_molecules_list)),
                        columns=['mol_id', 'PKI', 'conformations (ns)'] + list(range(0, num_columns)))

    for idx, dir_path in enumerate(sorted_folders): #dir_path = 0ns, 0.1ns, 0.2ns folder etc.
        print(dir_path.name)
        
        if os.path.isdir(dir_path):
            # List all files in the directory so 001.pdb, 002.pdb etc.
            files = os.listdir(dir_path)
            # Filter for PDB files
            pdb_files = [file for file in files if file.endswith('.pdb')]
            
            #create WHIM/GETAWAY vector for every molecule
            for idx, pdb_file in enumerate(pdb_files):
                pdb_file_path = os.path.join(dir_path, pdb_file)
                
                mol = Chem.MolFromPDBFile(pdb_file_path, removeHs=False, sanitize=False)
                
                if mol is not None:
                    # Sanitize the molecule
                    try:
                        Chem.SanitizeMol(mol)
                        #print("Molecule sanitized successfully.")
                    except ValueError as e:
                        print(f"Sanitization error: {e}")
                else:
                    print("not done molecule:")
                
                if descriptors == 'WHIM':
                    mol_descriptors = rdMolDescriptors.CalcWHIM(mol) #creates vector of the WHIM descriptors
                elif descriptors == 'GETAWAY':
                    mol_descriptors = rdMolDescriptors.CalcGETAWAY(mol)
                else:
                    print(f"Error: Choose a valid descriptor")
                
                #why 3? because '001.pdb' is the index, this takes '001' --> then int() gives 1
                index_to_insert = int(pdb_file[:3])+int((float(dir_path.name.rstrip('ns'))/10)*(len(sorted_folders)-1)*len(all_molecules_list))
                
                pki_value = molID_PKI_df.loc[molID_PKI_df['mol_id'] == int(pdb_file[:3]), 'PKI'].values[0]
                conformation_value = float(dir_path.name.rstrip('ns'))
                if conformation_value.is_integer():
                    conformation_value = int(conformation_value)
                total_df_conf_order.iloc[index_to_insert-1] = [int(pdb_file[:3]),pki_value,conformation_value] + mol_descriptors #why -1? to convert 001 to index 0 for example and list start at index 0.
        else:
            print('not a path')
    total_df_conf_order = total_df_conf_order.dropna().reset_index(drop=True)
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

# def csvfile_to_df(csvfile):
#     return df

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
    # for i, x in enumerate(np.arange(0, len(dfs) * timeinterval, timeinterval)):
    #     if x.is_integer():
    #         x = int(x)
    #     print(f"x: {x}, i: {i}")
    #     dfs[i].to_csv(final_path / f'{x}ns.csv', index=False)

#NOTE: this file does: get targets, count how many valid molecules and which,it creates the folder 'dataframes_WHIMJAK1' or equivellant
def main(ligand_conformations_path=public_variables.ligand_conformations_path_, \
         dataset_path=public_variables.dataset_csvfile_path_, \
            MDsimulations_path=public_variables.MDsimulations_path_,\
                descriptors=public_variables.RDKIT_descriptors_):

    #only contains molID and PKI value
    #NOTE: is it necessary to get the targets already?
    df_targets = get_targets(dataset_path) #df with mol_id and its PKI value

    #count how many directories there are in ligand_conformations_JAK1 (so how many time steps + rdkit one)
    x_ns_dirs = count_ns_dirs(ligand_conformations_path) #NOTE: not necessary??

    # #check how many invalids there are which we need to remove from the dataframes
    all_molecules_list, valid_mols, invalid_mols = trj_to_pdbfiles.get_molecules_lists(MDsimulations_path)
    print(all_molecules_list)
    #create the dataframes, which eventually will be placed in 'dataframes_JAK1_WHIM' and also add the targets to the dataframes.
    df_sorted_by_conf = create_full_dfs(ligand_conformations_path, df_targets, descriptors, all_molecules_list)
    df_sorted_by_molid = df_sorted_by_conf.sort_values(by=['mol_id', 'conformations (ns)']).reset_index(drop=True)
    public_variables.dataframes_master_.mkdir(parents=True, exist_ok=True)
    df_sorted_by_conf.to_csv(public_variables.dataframes_master_ / 'conformations_1000.csv', index=False)
    df_sorted_by_molid.to_csv(public_variables.dataframes_master_ / 'conformations_1000_moldid.csv', index=False)
    
    # # remove invalids
    # dic_with_dfs = remove_invalids_from_dfs(dfs_WHIM,invalid_mols)
    # save_dataframes(dic_with_dfs,public_variables.base_path_)
    return

if __name__ == "__main__":
    #x = prod_to_pdbfiles.main()

    ligand_conformations_path = public_variables.ligand_conformations_path_ # 'ligand_conformations_JAK1'
    dataset_csvfile_path = public_variables.dataset_csvfile_path_ # 'JAK1dataset.csv'
    MDsimulations_path = public_variables.MDsimulations_path_
    RDKIT_descriptors = public_variables.RDKIT_descriptors_

    main(ligand_conformations_path, dataset_csvfile_path, MDsimulations_path, RDKIT_descriptors)


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

