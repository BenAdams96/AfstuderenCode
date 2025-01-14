from global_files import public_variables

from pathlib import Path
import shutil
import subprocess
import pandas as pd
from io import StringIO
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
# import Afstuderen0.Afstuderen.removed.randomForest_add_MD_features as randomForest_add_MD_features
import MDAnalysis as mda
from MDAnalysis.coordinates import PDB
# import public_functions

import os
import re
import MDAnalysis as mda
import ast
# import MDAnalysis.analysis.psa

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    file_path = public_variables.base_path_ / 'code' / 'clustering' /  f'clustering_information_{public_variables.dataset_protein_}.csv'
    save_path = public_variables.base_path_ / 'code' / 'clustering' /  f'rmsd_clustersize_plot1.png'
    df = pd.read_csv(file_path)
    random_mol_ids = df['mol_id'].sample(n=3).unique() #get 3 random mol id
    # l = [1, 2, 3, 7, 8, 435, 356, 543]
    # # Create a plot
    plt.figure(figsize=(8, 6))

    # # Loop through the selected mol_ids and plot RMSD vs Cluster Size
    # for mol_id in l:
    #     mol_data = df[df['mol_id'] == mol_id]
    #     plt.plot(mol_data['RMSD'], mol_data['Number of clusters'], label=f'Mol ID {mol_id}')
    # Loop through the unique values in 'Number of Rotatable Bonds' (from 2 to 13)
    # Set up color palette for unique rotatable bonds
    colors = sns.color_palette("tab10", n_colors=11)  # Adjusts the number of distinct colors
    color_map = {num: colors[i] for i, num in enumerate(range(2, 13))}

    # Create a plot with a larger figure
    plt.figure(figsize=(10, 8))
    random_seed = 42
    # Loop through the unique values in 'Number of Rotatable Bonds' (from 2 to 13)
    for num_rot_bonds in range(2, 13):
        # Filter the DataFrame for the current number of rotatable bonds
        mol_data = df[df['Number of Rotatable Bonds'] == num_rot_bonds]
        
        # Check if there are at least 2 molecules with this number of rotatable bonds
        if len(mol_data) >= 2:
            # Randomly select two molecules
            random_mol_ids = mol_data['mol_id'].drop_duplicates().sample(n=2, random_state=random_seed).unique()
            
            # Plot RMSD vs Cluster Size for the selected molecules
            for mol_id in random_mol_ids:
                mol_id_data = mol_data[mol_data['mol_id'] == mol_id]
                plt.plot(mol_id_data['RMSD'],mol_id_data['Number of clusters'], 
                        label=f'mol_id: {mol_id} (Rot Bonds: {num_rot_bonds})', 
                        color=color_map[num_rot_bonds])
    # Adding labels and title
    plt.xlabel('RMSD')
    plt.ylabel('Number of clusters')
    plt.title('Number of clusters vs RMSD for Random Molecules')
    plt.legend()

    # Show the plot
    plt.savefig(save_path)


if __name__ == "__main__":
    main()