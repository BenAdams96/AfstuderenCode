# Standard library imports
import os
import re
import time

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed


# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolfiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Project-specific imports
from global_files import public_variables
import trj_to_pdbfiles

def generate_fingerprints_dataframe(smiles_tuple_list, radius=2, n_bits=2048):
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    data = []
    for str_idx, smiles in smiles_tuple_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = morgan_gen.GetFingerprint(Chem.MolFromSmiles(smiles))
            fp_list = [int(bit) for bit in fp.ToBitString()]
            data.append((int(str_idx), *fp_list))  # Store SMILES and fingerprint as a list

    # Create DataFrame
    df = pd.DataFrame(data, columns=["mol_id"] + [f"bit_{i}" for i in range(n_bits)])
    return df

def create_df_minimized(valid_mols_sorted, minimized_conformations_path):
    if public_variables.RDKIT_descriptors_ == 'WHIM':
        num_columns = 114
    elif public_variables.RDKIT_descriptors_ == 'GETAWAY':
        num_columns = 273
    else:
        raise ValueError("Error: Choose a valid descriptor")
    
    if public_variables.dataset_protein_ == 'JAK1':
        max_molecule_number = 615
    elif public_variables.dataset_protein_ == 'GSK3':
        max_molecule_number = 856

    rows = []

    for idx_str, pki in valid_mols_sorted: #loop over all valid molecules
        print(idx_str)
        dir_path = minimized_conformations_path / idx_str
        if os.path.isdir(dir_path): #enter the folder of '001' for example
            pdb_files = [file for file in os.listdir(dir_path) if file.endswith('.pdb')] #get all pdb files in this folder
            sorted_pdb_files = sorted(pdb_files, key=lambda x: int(x.split('conformation')[1].split('.pdb')[0]))
            for pdb_file in sorted_pdb_files:
                print(pdb_file)
                pdb_file_path = os.path.join(dir_path, pdb_file)
                mol = Chem.MolFromPDBFile(pdb_file_path, removeHs=False, sanitize=False)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                    except ValueError as e:
                        print(f"Sanitization error: {e}")
                        continue
                else:
                    print("Invalid molecule:")
                    print(pdb_file)
                    continue
                
                # Calculate descriptors
                if public_variables.RDKIT_descriptors_ == 'WHIM':
                    mol_descriptors = rdMolDescriptors.CalcWHIM(mol)
                elif public_variables.RDKIT_descriptors_ == 'GETAWAY':
                    mol_descriptors = rdMolDescriptors.CalcGETAWAY(mol)
                
                # index_to_insert = int(pdb_file[:3]) + int((float(dir_path.name.rstrip('ns')) / 10) * (len(sorted_folders) - 1) * len(all_molecules_list))
                
                conformation_value = int(pdb_file.split('conformation')[1].split('.pdb')[0])
                
                # Collect the row data
                rows.append([int(pdb_file[:3]), pki, conformation_value] + mol_descriptors)
        else:
            print('not a path')


    columns = ['mol_id', 'PKI', 'conformations (ns)'] + list(range(num_columns))
    total_df_conf_order = pd.DataFrame(rows, columns=columns).dropna().reset_index(drop=True)
    return total_df_conf_order

def main(dataset_path=public_variables.dataset_csvfile_path_):

    all_molecules_list, valid_mols, invalid_mols = trj_to_pdbfiles.get_molecules_lists(public_variables.MDsimulations_path_)
    valid_mols = set(valid_mols)
    print(valid_mols)
    fingerprint_df = pd.read_csv(dataset_path)
    
    molid_expmean_list = [(f"{mol_id:03d}", smiles) for mol_id, smiles in zip(fingerprint_df['mol_id'], fingerprint_df['exp_mean [nM]'])]
    validmolid__pki_list = [
        (item[0], -np.log10(item[1] * 1e-9))
        for item in molid_expmean_list
        if item[0] in valid_mols  # Filter based on valid_mols
    ]
    
    output_path = public_variables.base_path_ / Path('minimized_conformations')
    total_df_conf_order = create_df_minimized(validmolid__pki_list, output_path)
    print(type(output_path))
    total_df_conf_order.to_csv(output_path / '10minimized_conf.csv',index=False)

    return

if __name__ == "__main__":

    main()

    #get 20 minimized conformations
    #get 10 that are the least similar
    #use those as well for the models
    #conformations_10_minimized

