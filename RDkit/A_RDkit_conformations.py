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

# Project-specific imports
from global_files import public_variables
import trj_to_pdbfiles


def preprocess_molecule(smiles):
    """Convert SMILES to RDKit molecule and add hydrogens."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}")
        return None
    mol = Chem.AddHs(mol)
    return mol


def embed_and_minimize(mol, num_confs):
    """Generate 3D conformers and minimize energy for each."""
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)
    for i in range(num_confs):
        AllChem.MMFFOptimizeMolecule(mol, confId=i)
    return mol


def calculate_rmsd_matrix(mol, num_confs, str_idx):
    """Compute RMSD matrix for all pairs of conformations."""
    rmsd_matrix = np.zeros((num_confs, num_confs))
    for i in range(num_confs):
        print(f'conf {i}')
        for j in range(i + 1, num_confs):
            print(j)
            rmsd = AllChem.GetBestRMS(mol, mol, prbId=i, refId=j)
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd
    visualize_rmsd_matrix(rmsd_matrix, str_idx)
    return rmsd_matrix

def calculate_rmsd_pair(mol, i, j):
    """Compute RMSD for a specific pair of conformations."""
    print('printing a pair')
    return AllChem.GetBestRMS(mol, mol, prbId=i, refId=j)

def calculate_rmsd_matrix_parallel(mol, num_confs, str_idx):
    """Compute RMSD matrix for all pairs of conformations using parallel processing."""
    rmsd_matrix = np.zeros((num_confs, num_confs))
    
    pairs = [(i, j) for i in range(num_confs) for j in range(i + 1, num_confs)]
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(calculate_rmsd_pair)(mol, i, j) for i, j in pairs
    )
    
#     for (i, j), rmsd in zip(pairs, results):
#         rmsd_matrix[i, j] = rmsd
#         rmsd_matrix[j, i] = rmsd
    
#     visualize_rmsd_matrix(rmsd_matrix, str_idx)
#     return rmsd_matrix

def visualize_rmsd_matrix(rmsd_matrix, str_idx):
    """Generate a heatmap to visualize the RMSD matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(rmsd_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='RMSD (Å)')
    plt.title('RMSD Matrix of Conformations')
    plt.xlabel('Conformation Index')
    plt.ylabel('Conformation Index')
    plt.xticks(range(rmsd_matrix.shape[0]))
    plt.yticks(range(rmsd_matrix.shape[0]))
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(public_variables.base_path_ / 'minimized_conformations' / str_idx / 'rmsd_plot.png')

def select_most_distinct_conformations(rmsd_matrix, num_keep):
    """Select the most distinct conformations based on RMSD."""
    num_confs = rmsd_matrix.shape[0]
    selected_indices = [0]  # Start with the first conformation

    while len(selected_indices) < num_keep:
        max_rmsd = -1
        next_conformation_idx = -1
        for i in range(num_confs):
            if i not in selected_indices:
                min_rmsd_to_selected = min(rmsd_matrix[i, selected_indices])
                if min_rmsd_to_selected > max_rmsd:
                    max_rmsd = min_rmsd_to_selected
                    next_conformation_idx = i
        selected_indices.append(next_conformation_idx)

    return selected_indices


def save_conformations(mol, selected_indices, str_idx, output_dir):
    """Save selected conformations as PDB files."""
    pdb_files = []
    for i, conf_id in enumerate(selected_indices):
        pdb_file = output_dir /  f"{str_idx}_conformation{i+1}.pdb"
        rdmolfiles.MolToPDBFile(mol, pdb_file, confId=conf_id)
        pdb_files.append(pdb_file)
    return pdb_files


def generate_minimized_conformations(smiles_tuple, output_path, num_confs=20, num_keep=10):
    """Generate minimized conformations, compute RMSD, and select the most distinct."""
    str_idx, smiles = smiles_tuple
    print(f"Processing molecule {str_idx}")

    # Step 1: Preprocess molecule
    mol = preprocess_molecule(smiles)
    if mol is None:
        return []

    output_dir = Path(output_path) / str(str_idx)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Generate and minimize conformations
    mol = embed_and_minimize(mol, num_confs)
    print('embedded the molecules')
    # Step 3: Calculate RMSD matrix
    rmsd_matrix = calculate_rmsd_matrix(mol, num_confs, str_idx)
    print('calculated rmsd')
    # Step 4: Select the most distinct conformations
    selected_indices = select_most_distinct_conformations(rmsd_matrix, num_keep)
    print('selected indices')
    # Step 5: Save selected conformations to PDB files
    pdb_files = save_conformations(mol, selected_indices, str_idx, output_dir)

    return pdb_files


def main(dataset_path=public_variables.dataset_csvfile_path_):

    df = pd.read_csv(dataset_path)

    all_molecules_list, valid_mols, invalid_mols = trj_to_pdbfiles.get_molecules_lists(public_variables.MDsimulations_path_)

    smiles_list = [(f"{mol_id:03d}", smiles) for mol_id, smiles in zip(df['mol_id'], df['smiles'])]
    valid_smiles_list = [tuple for tuple in smiles_list if tuple[0] in valid_mols]

    # Ensure the output directory exists
    output_path = public_variables.base_path_ / 'minimized_conformations'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    start_time = time.time()
    for smiles in valid_smiles_list[40:615]:
        generate_minimized_conformations(smiles,output_path)
        elapsed_time = time.time() - start_time
        print(f"Time since last iteration: {elapsed_time:.2f} seconds")
        start_time = time.time()

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


    dataset_csvfile_path = public_variables.dataset_csvfile_path_ # 'JAK1dataset.csv'

    main(dataset_csvfile_path)

    #get 20 minimized conformations
    #get 10 that are the least similar
    #use those as well for the models
    #conformations_10_minimized

