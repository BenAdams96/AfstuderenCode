import numpy as np
from pathlib import Path

from global_files import public_variables
import pandas as pd
import math
import re

#NOTE: takes in a folder with csv files 'dataframes_WHIMJAK1' (so 0ns.csv, 1ns.csv) and will convert it to a list of dataframes.
# so, takes in 'dataframes_WHIMJAK1' or 'dataframes_WHIMJAK1_with0.85', and creates 
def main(folder_name):
    base_path = Path(__file__).resolve().parent
    dfs_path = Path(base_path) / public_variables.dfs_descriptors_only_path_
    print(dfs_path)
    print('hi')
    dfs = csvfiles_to_dfs(dfs_path)
    return dfs

def csvfiles_to_dfs(dfs_path):
    '''The folder with CSV files 'dataframes_JAK1_WHIM' is the input and these CSV files will be
       put into a list of pandas DataFrames, also works with 0.5 1 1.5 formats.
    '''
    dfs = []
    # Define the regex pattern to match filenames like '0ns.csv', '1ns.csv', '1.5ns.csv', etc.
    pattern = re.compile(r'^\d+(\.\d+)?ns\.csv$')

    for csv_file in sorted(dfs_path.glob('*.csv'), key=lambda x: extract_number(x.name)):
        print(csv_file)
        if pattern.match(csv_file.name):  # Check if the file name matches the pattern
            print(f"Reading {csv_file}")
            # Read CSV file into a DataFrame and append to the list
            dfs.append(pd.read_csv(csv_file))
    return dfs

# def csvfile_to_df(csvfile):

#     return df

def extract_number(filename):
    # Use regular expression to extract numeric part (integer or float) before 'ns.csv'
    match = re.search(r'(\d+(\.\d+)?)ns\.csv', filename)
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
    

if __name__ == "__main__":
    main()
