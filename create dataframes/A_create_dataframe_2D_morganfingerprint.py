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


def main(dataset_path=public_variables.dataset_path_):

    fingerprint_df = pd.read_csv(dataset_path)
    print(fingerprint_df)
    all_molecules_list, valid_mols, invalid_mols = trj_to_pdbfiles.get_molecules_lists(public_variables.MDsimulations_path_)
    
    smiles_list = [(f"{mol_id:03d}", smiles) for mol_id, smiles in zip(fingerprint_df['mol_id'], fingerprint_df['smiles'])]
    targets_df = -np.log10(fingerprint_df['exp_mean [nM]'] * 1e-9)
    fingerprint_df['mol_id'] = fingerprint_df['mol_id'].apply(lambda x: f"{x:03d}")
    targets_df_valid = targets_df[fingerprint_df['mol_id'].isin(valid_mols)]
    
    valid_smiles_list = [tuple for tuple in smiles_list if tuple[0] in valid_mols]

    # Ensure the output directory exists
    dataframes_master_ = public_variables.base_path_ / Path(f'2D_models')
    output_path =  dataframes_master_ / '2D_ECFP.csv'

    if not os.path.exists(dataframes_master_):
        os.makedirs(dataframes_master_)

    start_time = time.time()

    fingerprint_df = generate_fingerprints_dataframe(valid_smiles_list, radius=2, n_bits=2048)
    fingerprint_df.insert(1, 'PKI', targets_df_valid.values)
    fingerprint_df.to_csv(output_path, index=False)
    # #only contains molID and PKI value
    # df_targets = get_targets(dataset_path) #df with columns: 'mol_id' and 'PKI value'. all molecules

    # # #check how many invalids there are which we need to remove from the dataframes
    # all_molecules_list, valid_mols, invalid_mols = trj_to_pdbfiles.get_molecules_lists(MDsimulations_path)
    # print(all_molecules_list)
    # #create the dataframes, which eventually will be placed in 'dataframes_JAK1_WHIM' and also add the targets to the dataframes.
    # df_sorted_by_configuration = create_full_dfs(ligand_conformations_path, df_targets, descriptors, all_molecules_list)
    # df_sorted_by_molid = df_sorted_by_configuration.sort_values(by=['mol_id', 'conformations (ns)']).reset_index(drop=True)

    # public_variables.dataframes_master_.mkdir(parents=True, exist_ok=True)
    # df_sorted_by_configuration.to_csv(public_variables.initial_dataframe, index=False)
    # df_sorted_by_molid.to_csv(public_variables.dataframes_master_ / 'initial_dataframe_mol_id.csv', index=False)
    return

if __name__ == "__main__":


    dataset_csvfile_path = public_variables.dataset_path_ # 'JAK1dataset.csv'

    main(dataset_csvfile_path)

    #get 20 minimized conformations
    #get 10 that are the least similar
    #use those as well for the models
    #conformations_10_minimized

