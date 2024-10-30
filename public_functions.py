from pathlib import Path
from public_variables import features
import shutil
import subprocess
import pandas as pd
from io import StringIO
import public_variables
import os


def get_molecules_lists(MDsimulations_path):
    '''uses the MD_simulations folder and checks for every molecule the simulation whether it contains the .tpr and .xtc file
        to see if it contains a valid trajectory file (.tpr)
        output: lists of all the molecules, valid molecules, and invalid molecules in strings
        '''
    print('get molecules list, valid_mols, invalid_mols')
    molecules_list = []
    valid_mols = []
    invalid_mols = []
    for item in MDsimulations_path.iterdir(): #item is every folder '001' etc in the MD folder
        xtcfile = f"{item.name}_prod.xtc"  # trajectory file
        trajectory_file = item / xtcfile

        if item.is_dir():  # Check if the item is a directory
            molecules_list.append(item.name)
        if not trajectory_file.exists():
            invalid_mols.append(item.name)
        else:
            valid_mols.append(item.name)

    return molecules_list, valid_mols, invalid_mols #['001','002']

def main(base_path):
    
    return

if __name__ == "__main__":
    base_path = public_variables.base_path_
    main(base_path)

