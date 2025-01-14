from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import itertools
from typing import List
import numpy as np
from pathlib import Path
import pandas as pd
import math
import re
import os
from global_files import public_variables

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdForceFieldHelpers



# input_directory = Path(__file__).resolve().parent / 'pdb'
# output_directory = public
# output_directory.mkdir(exist_ok=True)

# for pdb_file in input_directory.glob('*.pdb'):
#     try:
#         # Load the molecule from the PDB file
#         mol = Chem.MolFromPDBFile(str(pdb_file), removeHs=False)

#         # Check if the molecule is successfully loaded
#         if mol is None:
#             print(f"Could not load {pdb_file}. Skipping...")
#             continue

#         # Add hydrogens to the molecule
#         mol = Chem.AddHs(mol)

#         # Generate 3D coordinates if not already present
#         if mol.GetNumConformers() == 0:
#             rdDistGeom.EmbedMolecule(mol)

#         # Optimize the 3D geometry using the UFF force field
#         rdForceFieldHelpers.UFFOptimizeMolecule(mol)

#         filename_stem = pdb_file.stem
#         padded_number = f"{int(filename_stem):03}"

#         # Create the new filename with the padded number
#         new_filename = padded_number + pdb_file.suffix
        
#         print(new_filename)
#         # Define the output file path
#         output_file = output_directory / new_filename

#         # Save the optimized structure to the output PDB file
#         Chem.MolToPDBFile(mol, str(output_file))
        
#         print(f"Optimized structure saved to {output_file}")

#     except Exception as e:
#         print(f"An error occurred with {pdb_file}: {e}")

def add_RDkit_data(pdb_directory, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    for pdb_file in pdb_directory.glob('*.pdb'):
        try:
            # Load the molecule from the PDB file
            mol = Chem.MolFromPDBFile(str(pdb_file), removeHs=False)

            # Check if the molecule is successfully loaded
            if mol is None:
                print(f"Could not load {pdb_file}. Skipping...")
                continue

            # Add hydrogens to the molecule
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates if not already present
            if mol.GetNumConformers() == 0:
                print('!!!!!! NO 3D structure !!!!!!')
                rdDistGeom.EmbedMolecule(mol)

            # Optimize the 3D geometry using the MMFF force field
            rdForceFieldHelpers.MMFFOptimizeMolecule(mol)

            filename_stem = pdb_file.stem
            padded_number = f"{int(filename_stem):03}"

            # Create the new filename with the padded number
            new_filename = padded_number + pdb_file.suffix
            
            print(new_filename)
            # Define the output file path
            output_file = output_directory / new_filename

            # Save the optimized structure to the output PDB file
            Chem.MolToPDBFile(mol, str(output_file))
            
            print(f"Optimized structure saved to {output_file}")

        except Exception as e:
            print(f"An error occurred with {pdb_file}: {e}")
    return

def main():
    base_path = public_variables.base_path_
    input_directory = base_path / 'pdb'
    output_directory = base_path / public_variables.lig_conforms_foldername_ / 'rdkit_min'
    add_RDkit_data(input_directory, output_directory)
    return

if __name__ == "__main__":
    main()