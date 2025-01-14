import math
import numpy as np
import os
import MDAnalysis as mda
from MDAnalysis.coordinates import PDB
import rdkit
from global_files import public_variables
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdMolDescriptors
import trj_to_pdbfiles
import pandas as pd
from pathlib import Path
import re
import pathlib

def remove_constant_columns(df):
    # Identify constant columns
    constant_columns = df.columns[df.nunique() <= 1]
    
    if not constant_columns.empty:
        print(f"The following constant columns were removed: {', '.join(constant_columns)}")
    
    # Remove constant columns and keep only non-constant columns
    non_constant_columns = df.loc[:, df.nunique() > 1]
    return non_constant_columns

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

def main():
    df = pd.read_csv(public_variables.dataset_path_)
    df['mol_id'] = range(1, len(df) + 1)
    df.to_csv(public_variables.base_path_ / public_variables.dataset_filename_, index=False)
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

