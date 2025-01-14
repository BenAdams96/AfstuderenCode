from pathlib import Path
import shutil
import subprocess
import pandas as pd
from io import StringIO
from global_files import public_variables, public_functions
import os
import re

def get_edr_files(MDsimulations_path, edrfolder_path):
    '''create the edr folder in the folder 'energyfolder_files' and get the edr files from the MDsimulations
    '''
    target_path = edrfolder_path
    target_path.mkdir(parents=True, exist_ok=True)
    for folder in MDsimulations_path.iterdir():
        if folder.is_dir():  # Check if it is a directory
            # Find the .edr file in the current folder
            for edr_file in folder.glob('*prod.edr'):
                destination_path = target_path / edr_file.name
                shutil.copy(edr_file, destination_path) # Copy the .edr file to the target directory

                # #for if we want the names to be 001.edr
                # new_name = edr_file.name.replace('_prod', '')
                # new_destination = target_path / new_name
                # shutil.copy(edr_file, new_destination)
    return

def make_xvg_files(edrfolder_path, xvgfolder_path, features, valid_mols):
    ''' create the xvgfolder, add prefixes to the valid molecules if they are missing
        for every valid molecule, run the gmx energy command
    '''
    if xvgfolder_path.exists():
        i = 1
        # Create a new path with the suffix
        new_xvgfolder_path = xvgfolder_path.parent / f"{xvgfolder_path.name}_v{i}"
        # Keep incrementing the suffix until we find a non-existing path
        while new_xvgfolder_path.exists():
            i += 1
            new_xvgfolder_path = xvgfolder_path.parent / f"{xvgfolder_path.name}_v{i}"
        # Update the xvgfolder_path to the new path
        xvgfolder_path = new_xvgfolder_path
        public_variables.xvgfolder_path_ = xvgfolder_path        
        
    xvgfolder_path.mkdir(parents=True, exist_ok=True)
    padded_valid_mols = [number.zfill(3) for number in valid_mols]
    
    for mol in padded_valid_mols:
        run_gmx_energy(edrfolder_path, xvgfolder_path, mol, features)
    return xvgfolder_path

def run_gmx_energy(edrfolder_path, xvgfolder_path, molecule, features):
    ''' create the 'xvg_files' folder to store the xvg files
        run the gmx energy command for every molecule
        !!! NEEDS TO BE RUN ON LINUX/BASH !!!
    '''

    # create the path for the xvg file to be stored
    xvg_file = xvgfolder_path / f'{molecule}.xvg'
    
    # If the file already exists, modify the filename to avoid overwriting
    if xvg_file.exists():
        # Generate a new folder for the xvg files with a suffix to avoid overwriting
        i = 1
        xvgfolder_path = xvgfolder_path / f'_{i}.xvg'
        while xvgfolder_path.exists():
            i += 1
    
    # Construct the command with the variable file prefix and output directory
    command = f"gmx energy -f {molecule}_prod.edr -o {xvg_file}"

    # specify the user input which are the features
    user_input = '\n'.join(features)
    
    if edrfolder_path.exists() and edrfolder_path.is_dir():
        os.chdir(edrfolder_path)

    #run the command which creates the xvg files
    subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)

def generate_new_csv_filename(xvgfolder_path: Path, csv_filename: str) -> str:
    """ make sure that MDfeatures_allmol_csvfile gets a new name if the csv file already exists and
        thus corresponds with the correct xvg file"""
    # Convert the PosixPath to a string
    xvgfolder_str = str(xvgfolder_path)

    # Extract the number after the last underscore if it exists
    match = re.search(r'_v(\d+)$', xvgfolder_str)
    
    # Check if there's a match for "_number" at the end of the string
    if match:
        print('if')
        xvg_file_number = match.group(1)
        # Insert the extracted number before '.csv'
        new_csv_filename = csv_filename.replace('.csv', f'_v{xvg_file_number}.csv')
    else:
        print('else')
        # If no match, keep the original filename
        new_csv_filename = csv_filename

    return new_csv_filename

def xvg_files_to_csvfile(energyfolder_path, xvgfolder_path):
    #TODO: make this correct
    ''' go over all the xvg files in chronical order and create one big csv file which 
    '''
    all_data = pd.DataFrame()
    #TODO: xvgfolder_path doesnt use the correct folderpath yet. it uses the 'xvg_files' everytime
    
    for xvg_file in sorted(xvgfolder_path.glob('*.xvg')):   #go over the xvg files/molecules in chronical order
        column_names = ['picoseconds']
        # Read the file and filter out lines starting with '#' or '@', while extracting legends
        with open(xvg_file, 'r') as file:
            lines = file.readlines()
            data_lines = []
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    if line.startswith('@ s'):
                        # Extract the legend name
                        column_name = line.split('"')[1] #get "Bond" "Enthalpy" etc
                        column_names.append(column_name)
                else:
                    data_lines.append(line)
        
        # Join the filtered lines into a single string
        data_str = ''.join(data_lines) #this is the complete xvg file but without the first 30 lines/comments or so

        # Use StringIO to read the string as if it were a file
        data = pd.read_csv(StringIO(data_str), sep='\s+', header=None)

        #NOTE: xvg file stays of equal length (independent of timeinterval_snapshot) so variable data as well
        #NOTE: variable 'filtered_dataframe' only contains the dataline at the time intervals we are interested in
        filtered_dataframe = data
        # Rename the columns with the extracted legends
        filtered_dataframe.columns = column_names
        # add a column with the mol id
        filtered_dataframe.insert(0, 'mol_id', xvg_file.name.rsplit('.', 1)[0])
        
        # Append the selected columns to the all_data DataFrame
        all_data = pd.concat([all_data, filtered_dataframe], ignore_index=True)
    # concatenate the dataframes under eachother and save them in a csv file
    #
    
    # MDfeatures_allmol_csvfile_ = generate_new_csv_filename(xvgfolder_path, public_variables.MDfeatures_allmol_csvfile)
    all_data.to_csv(energyfolder_path / public_variables.MDfeatures_allmol_csvfile, index=False)
    return




def main(MDsimulations_path_):
    base_path = public_variables.base_path_
    MDsimulations_path = MDsimulations_path_
    

    # get all the edr files
    energyfolder_path = public_variables.energyfolder_path_
    edrfolder_path = public_variables.edrfolder_path_
    xvgfolder_path = public_variables.xvgfolder_path_
    get_edr_files(MDsimulations_path, edrfolder_path)

    #use the edr files to create xvg files
    valid_mols = public_functions.get_molecules_lists(MDsimulations_path)[1]
    
    #create xvg files from the edr files, check public variables which features we want
    features = public_variables.features
    new_xvgfolder_path = make_xvg_files(edrfolder_path, xvgfolder_path, features, valid_mols)
    
    #use the created xvg files to create csv dataframes
    xvg_files_to_csvfile(energyfolder_path, new_xvgfolder_path)
    return

if __name__ == "__main__":
    #NOTE: ONLY NEEDS MDSIMULATION path/folder
    MDsimulations_path = public_variables.MDsimulations_path_
    main(MDsimulations_path)

