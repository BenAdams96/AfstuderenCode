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
import numpy as np
from pathlib import Path

import glob
import public_variables
import pandas as pd
import math
import re

#NOTE: takes in a folder with csv files 'dataframes_WHIMJAK1' (so 0ns.csv, 1ns.csv) and will convert it to a dictionary with keys the foldername and value a panda dataframe
# so, takes in 'dataframes_WHIMJAK1' or 'dataframes_WHIMJAK1_with0.85', and creates 
def main(folder_name = public_variables.dfs_descriptors_only_path_, exclude_files=None):
    base_path = public_variables.base_path_
    dfs_path = Path(base_path) / folder_name
    
    dic = csvfiles_to_dic(dfs_path, exclude_files)
    return dic

def csvfiles_to_dic(dfs_path, exclude_files: list = []):
    '''The folder with CSV files 'dataframes_JAK1_WHIM' is the input and these CSV files will be
       put into a list of pandas DataFrames, also works with 0.5 1 1.5 formats.
    '''
    if exclude_files is None:
        exclude_files = []
    dic = {}
    for csv_file in dfs_path.glob('*.csv'):
        # print(csv_file.name)
        if csv_file.name not in exclude_files:
            dic[csv_file.stem] = pd.read_csv(csv_file)
        else:
            continue
            # print(f'name {csv_file} is in exclude')
    # Define the regex pattern to match filenames like '0ns.csv', '1ns.csv', '1.5ns.csv', etc.
    # pattern = re.compile(r'^\d+(\.\d+)?ns\.csv$')

    # for csv_file in sorted(dfs_path.glob('*.csv'), key=lambda x: extract_number(x.name)):
    #     print(csv_file)
    #     if pattern.match(csv_file.name):  # Check if the file name matches the pattern
    #         print(f"Reading {csv_file}")
    #         # Read CSV file into a DataFrame and append to the list
    #         dfs.append(pd.read_csv(csv_file))
    return dic

# def csvfile_to_df(csvfile):
#     return df

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

# # def csvfile_to_df(csvfile):
# #     return df

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


def get_sorted_folders_namelist(file_list):
    '''
    This function takes a list of CSV filenames and returns a sorted list of filenames.
    Files with numeric values before 'ns.csv' are sorted based on these values.
    Files without 'ns.csv' are placed at the beginning of the list.
    '''
    # Define the regex pattern to match filenames like '0ns.csv', '1ns.csv', '1.5ns.csv', etc.
    pattern = re.compile(r'^(\d+(\.\d+)?)ns$')

    # Sort the files based on the extracted number or position them at the beginning
    sorted_files = sorted(file_list, key=lambda x: extract_number2(x, pattern))

    return sorted_files

def extract_number2(filename, pattern):
    '''
    Extracts the numeric value from filenames matching the pattern before 'ns.csv'.
    Returns float('inf') for filenames that do not match the pattern.
    '''
    match = pattern.search(filename)
    if match:
        number_str = match.group(1)
        number = float(number_str)
        return number
    else:
        return float('-inf')
    

def get_sorted_columns(column_list):
    """
    This function takes a list of column names and returns a sorted list.
    Columns with 'ns' are sorted numerically, followed by columns starting with 'conformations'
    sorted by the numeric value after 'conformations_'.
    """
    # Define the regex pattern for matching 'ns' columns and 'conformations' columns
    ns_pattern = re.compile(r'^(\d+(\.\d+)?)ns$')
    conformations_pattern = re.compile(r'^conformations_(\d+)$')

    # Separate 'ns' columns and 'conformations' columns
    ns_columns = [col for col in column_list if ns_pattern.match(col)]
    conformations_columns = [col for col in column_list if conformations_pattern.match(col)]

    # Sort 'ns' columns based on the numeric value before 'ns'
    sorted_ns = sorted(ns_columns, key=lambda x: float(x[:-2]))  # Remove 'ns' and convert to float

    # Sort 'conformations' columns based on the numeric value after 'conformations_'
    sorted_conformations = sorted(conformations_columns, key=lambda x: int(x.split('_')[1]))

    # Combine sorted lists
    sorted_columns = sorted_ns + sorted_conformations

    return sorted_columns
    

if __name__ == "__main__":
    
    main()
