# Standard library imports
import os
import re

# Third-party libraries
import numpy as np
import pandas as pd

# RDKit imports
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Project-specific imports
import public_variables
import trj_to_pdbfiles


def get_targets(dataset):
    """read out the original dataset csv file and get the targets + convert exp_mean to PKI
    out: pandas dataframe with two columns: ['mol_id','PKI']
    """
    df = pd.read_csv(dataset)
    df['PKI'] = -np.log10(df['exp_mean [nM]'] * 1e-9)
    return df[['mol_id','PKI']]
    
def create_full_dfs(ligand_conformations_path, molID_PKI_df, descriptors, valid_molecules):
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
    
    sorted_folders = get_sorted_folders(ligand_conformations_path)  # Sorted from 0ns to 10ns
    filtered_paths = [path for path in sorted_folders if float(path.name.replace('ns', '')) * 10 % 1 == 0] #only use stepsize of 0.1 instead of 0.01
    print('valid molecule')
    print(valid_molecules)
    rows = []

    for idx, dir_path in enumerate(filtered_paths):  # dir_path = 0ns, 0.1ns, 0.2ns folder etc.
        print(dir_path.name)
        
        if os.path.isdir(dir_path):
            # List all PDB files in the directory
            pdb_files = [file for file in os.listdir(dir_path) if file.endswith('.pdb')]
            filtered_sorted_list = sorted([file for file in pdb_files if int(file.split('_')[0]) <= max_molecule_number], #TODO: shitty solution
                              key=lambda x: int(x.split('_')[0]))
            
            # print(filtered_sorted_list)
            for pdb_file in filtered_sorted_list:
                
                pdb_file_path = os.path.join(dir_path, pdb_file)
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
                conformation_value = float(dir_path.name.rstrip('ns'))
                if conformation_value.is_integer():
                    conformation_value = int(conformation_value)
                
                # Collect the row data
                rows.append([int(pdb_file[:3]), pki_value, conformation_value] + mol_descriptors)
        else:
            print('not a path')

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

#NOTE: this file does: get targets, count how many valid molecules and which,it creates the folder 'dataframes_WHIMJAK1' or equivellant
def main(ligand_conformations_path=public_variables.ligand_conformations_path_, \
         dataset_path=public_variables.dataset_csvfile_path_, \
            MDsimulations_path=public_variables.MDsimulations_path_,\
                descriptors=public_variables.RDKIT_descriptors_):

    #only contains molID and PKI value
    #NOTE: is it necessary to get the targets already?
    df_targets = get_targets(dataset_path) #df with columns: 'mol_id' and 'PKI value'. all molecules

    # #check how many invalids there are which we need to remove from the dataframes
    all_molecules_list, valid_mols, invalid_mols = trj_to_pdbfiles.get_molecules_lists(MDsimulations_path)
    
    #create the dataframes, which eventually will be placed in 'dataframes_JAK1_WHIM' and also add the targets to the dataframes.
    df_sorted_by_configuration = create_full_dfs(ligand_conformations_path, df_targets, descriptors, valid_molecules=valid_mols)
    df_sorted_by_molid = df_sorted_by_configuration.sort_values(by=['mol_id', 'conformations (ns)']).reset_index(drop=True)

    public_variables.dataframes_master_.mkdir(parents=True, exist_ok=True)
    df_sorted_by_configuration.to_csv(public_variables.initial_dataframe, index=False)
    df_sorted_by_molid.to_csv(public_variables.dataframes_master_ / 'initial_dataframe_mol_id.csv', index=False)
    return

if __name__ == "__main__":
    ligand_conformations_path = public_variables.ligand_conformations_path_ # 'ligand_conformations_JAK1'
    dataset_csvfile_path = public_variables.dataset_csvfile_path_ # 'JAK1dataset.csv'
    MDsimulations_path = public_variables.MDsimulations_path_
    RDKIT_descriptors = public_variables.RDKIT_descriptors_

    main(ligand_conformations_path, dataset_csvfile_path, MDsimulations_path, RDKIT_descriptors)

