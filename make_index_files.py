
import public_variables
from pathlib import Path
import shutil
import subprocess
import pandas as pd
from io import StringIO
# import Afstuderen0.Afstuderen.removed.randomForest_add_MD_features as randomForest_add_MD_features
import public_variables
# import public_functions

import os
import re
import MDAnalysis as mda
# import MDAnalysis.analysis.psa

import numpy as np
from pathlib import Path

# def make_index_file2(path, pdb_filename):
#     ''' create the 'xvg_files' folder to store the xvg files
#         run the gmx energy command for every molecule
#         !!! NEEDS TO BE RUN ON LINUX/BASH !!!
#     '''

#     # create the path for the xvg file to be stored
#     xvg_file = xvgfolder_path / f'{molecule}.xvg'

#     # If the file already exists, modify the filename to avoid overwriting
#     if xvg_file.exists():
#         # Generate a new folder for the xvg files with a suffix to avoid overwriting
#         i = 1
#         xvgfolder_path = xvgfolder_path / f'_{i}.xvg'
#         while xvgfolder_path.exists():
#             i += 1

#     # Construct the command with the variable file prefix and output directory
#     command = f"gmx energy -f {molecule}_prod.edr -o {xvg_file}"

#     # specify the user input which are the features
#     user_input = '\n'.join(features)

#     if edrfolder_path.exists() and edrfolder_path.is_dir():
#         os.chdir(edrfolder_path)

#     #run the command which creates the xvg files
#     subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)

def make_PSA_index_files(MD_path):
    os.chdir(MD_path)
    for padded_num in (f"{i:03}" for i in range(1,857)): #not abstract yet, not necessary
        pdb_file = MD_path / padded_num / f'{padded_num}_prod.pdb'
        ndx_file = MD_path / padded_num / f'index_PSA_{padded_num}.ndx'
        os.chdir(MD_path / padded_num)
        if pdb_file.exists():
            # Construct the command and its arguments as a list
            command = ["gmx", "make_ndx", "-f", str(pdb_file), "-o", str(ndx_file)]
            
            # Run the command using subprocess and provide inputs interactively
            try:
                with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
                    # Provide the inputs to the command
                    proc.communicate(input='r 1 & a N* O* S*\nq\n')
                print(f"Index file created: {ndx_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running the command: {e}")
            except FileNotFoundError as e:
                print(f"Command not found: {e}")
        else:
            print(f'tpr not present in: {padded_num}')
            continue
    return

def run_gmx_psa_sasa(MD_path, output_path):
    ''' create the 'xvg_files' folder to store the xvg files
        run the gmx energy command for every molecule
        !!! NEEDS TO BE RUN ON LINUX/BASH !!!
    '''
    output_path.mkdir(parents=True, exist_ok=True)
    # create the path for the xvg file to be stored
    for padded_num in (f"{i:03}" for i in range(1, 857)):
        os.chdir(MD_path / f'{padded_num}')
        sasa_file = output_path / f'{padded_num}.xvg'
        xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        ndx_file = MD_path / padded_num / f'index_PSA_{padded_num}.ndx'
        # If the file already exists, modify the filename to avoid overwriting
        if tpr_file.exists():
        # Generate a new folder for the xvg files with a suffix to avoid overwriting
            command = f'gmx sasa -f {xtc_file} -s {tpr_file} -n {ndx_file} -o {sasa_file}'
            user_input = '6'
            #run the command which creates the xvg files
            subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)
        else:
            continue
    return

def psa_xvg_files_to_csvfiles(energyfolder_path, sasa_xvgfolder_path): # for PSA
    ''' go over all the xvg files in chronical order and create one big csv file which 
    '''
    all_data = pd.DataFrame()
    
    for xvg_file in sorted(sasa_xvgfolder_path.glob('*.xvg')):   #go over the xvg files/molecules in chronical order
        print(xvg_file)
        column_names = ['picoseconds']
        # Read the file and filter out lines starting with '#' or '@', while extracting legends
        with open(xvg_file, 'r') as file:
            lines = file.readlines()
            data_lines = []
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    if line.startswith('@ s'):
                        # Extract the legend name
                        column_name = "PSA"
                        column_names.append(column_name)
                else:
                    data_lines.append(line)
        
        # Join the filtered lines into a single string
        data_str = ''.join(data_lines) #this is the complete xvg file but without the first 30 lines/comments or so

        # Use StringIO to read the string as if it were a file
        data = pd.read_csv(StringIO(data_str), sep='\s+', header=None)

        #NOTE: xvg file stays of equal length (independent of timeinterval_snapshot) so variable data as well
        #NOTE: variable 'filtered_dataframe' only contains the dataline at the time intervals we are interested in
        filtered_dataframe = data #data[data[0] % (public_variables.timeinterval_snapshots*1000) == 0]
        # Rename the columns with the extracted legends
        filtered_dataframe.columns = column_names
        # add a column with the mol id
        filtered_dataframe.insert(0, 'mol_id', xvg_file.name.rsplit('.', 1)[0])
        
        # Append the selected columns to the all_data DataFrame
        all_data = pd.concat([all_data, filtered_dataframe], ignore_index=True)
        
    # concatenate the dataframes under eachother and save them in a csv file
    #
    
    #MDfeatures_allmol_csvfile_ = generate_new_csv_filename(sasa_xvgfolder_path, public_variables.MDfeatures_allmol_csvfile)
    all_data.to_csv(energyfolder_path / 'psa.csv', index=False)
    return


def run_gmx_dipoles(MD_path, output_path_TDM, output_path_epsilon):
    ''' 
    Create the 'xvg_files' folder to store the xvg files.
    Run the gmx dipoles command for every molecule.
    !!! NEEDS TO BE RUN ON LINUX/BASH !!!
    '''
    output_path_TDM.mkdir(parents=True, exist_ok=True)  # Create the folder for storing xvg files
    output_path_epsilon.mkdir(parents=True, exist_ok=True)
    # Loop over the molecules (using padded numbers)
    for padded_num in (f"{i:03}" for i in range(1, 857)):
        
        os.chdir(MD_path / f'{padded_num}')
        dipole_file = output_path_TDM / f'{padded_num}.xvg'
        epsilon_file = output_path_epsilon / f'{padded_num}.xvg'
        xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'

        # If the .tpr file exists, run the gmx dipoles command
        if tpr_file.exists():
            
            # Construct the GROMACS dipoles command
            command = f'gmx dipoles -f {xtc_file} -s {tpr_file} -n {ndx_file} -o {dipole_file} -eps {epsilon_file}'
            user_input = '2'  # Select the appropriate group (adjust if necessary)
            
            # Run the command to calculate the dipole moment
            subprocess.run(command, shell=True, input=user_input, capture_output=True, text=True)
        else:
            print(padded_num)
            continue

    return

def Total_dipole_moment_xvg_files_to_csvfiles(energyfolder_path, totaldipoleMoment_xvgfolder_path):
    ''' Go over all the xvg files in chronological order and create one big CSV file. 
        For .xvg files with multiple columns, only keep the first (picoseconds) and second columns.
    '''
    all_data = pd.DataFrame()  # Initialize an empty DataFrame
    
    for xvg_file in sorted(totaldipoleMoment_xvgfolder_path.glob('*.xvg')):   # Go over the xvg files in chronological order
        print(xvg_file)
        column_names = ['picoseconds', 'Mtot']  # Adjust column names for the first and second columns

        # Read the file and filter out lines starting with '#' or '@', while extracting legends
        with open(xvg_file, 'r') as file:
            lines = file.readlines()
            data_lines = []
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    continue  # Skip comment and metadata lines
                else:
                    data_lines.append(line)  # Collect data lines
        
        # Join the filtered lines into a single string
        data_str = ''.join(data_lines)  # This is the complete xvg file but without comments

        # Use StringIO to read the string as if it were a file
        data = pd.read_csv(StringIO(data_str), sep='\s+', header=None)

        # Select only the first (picoseconds) and second columns (SASA)
        data = data[[0, 4]]

        # Filter the data to match the time intervals you are interested in
        filtered_dataframe = data #data[data[0] % (public_variables.timeinterval_snapshots * 1000) == 0]

        # Rename the columns
        filtered_dataframe.columns = column_names

        # Add a column with the molecule ID (from the filename)
        filtered_dataframe.insert(0, 'mol_id', xvg_file.name.rsplit('.', 1)[0])

        # Append the selected columns to the all_data DataFrame
        all_data = pd.concat([all_data, filtered_dataframe], ignore_index=True)
        #print(all_data)

    # Save the concatenated data as a CSV file
    all_data.to_csv(energyfolder_path / 'dipole_moment_total.csv', index=False)
    
    return


def epsilon_xvg_files_to_csvfiles(energyfolder_path, epsilon_xvgfolder_path):
    ''' Go over all the xvg files in chronological order and create one big CSV file. 
        For .xvg files with multiple columns, only keep the first (picoseconds) and second columns.
    '''
    all_data = pd.DataFrame()  # Initialize an empty DataFrame
    
    for xvg_file in sorted(epsilon_xvgfolder_path.glob('*.xvg')):   # Go over the xvg files in chronological order
        print(xvg_file)
        column_names = ['picoseconds', 'epsilon']  # Adjust column names for the first and second columns

        # Read the file and filter out lines starting with '#' or '@', while extracting legends
        with open(xvg_file, 'r') as file:
            lines = file.readlines()
            data_lines = []
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    continue  # Skip comment and metadata lines
                else:
                    data_lines.append(line)  # Collect data lines
        
        # Join the filtered lines into a single string
        data_str = ''.join(data_lines)  # This is the complete xvg file but without comments

        # Use StringIO to read the string as if it were a file
        data = pd.read_csv(StringIO(data_str), sep='\s+', header=None)

        # Select only the first (picoseconds) and second columns (SASA)
        data = data[[0, 1]]

        # Filter the data to match the time intervals you are interested in
        filtered_dataframe = data #data[data[0] % (public_variables.timeinterval_snapshots * 1000) == 0]

        # Rename the columns
        filtered_dataframe.columns = column_names

        # Add a column with the molecule ID (from the filename)
        filtered_dataframe.insert(0, 'mol_id', xvg_file.name.rsplit('.', 1)[0])

        # Append the selected columns to the all_data DataFrame
        all_data = pd.concat([all_data, filtered_dataframe], ignore_index=True)
        #print(all_data)

    # Save the concatenated data as a CSV file
    all_data.to_csv(energyfolder_path / 'epsilon.csv', index=False)
    
    return

def run_gmx_gyrate(MD_path, output_path):
    ''' 
    Create the 'xvg_files' folder to store the xvg files.
    Run the gmx gyrate command for every molecule.
    !!! NEEDS TO BE RUN ON LINUX/BASH !!!
    '''
    output_path.mkdir(parents=True, exist_ok=True)  # Create the folder for storing xvg files

    # Loop over the molecules (using padded numbers)
    for padded_num in (f"{i:03}" for i in range(1, 857)):
        os.chdir(MD_path / f'{padded_num}')
        gyrate_file = output_path / f'{padded_num}.xvg'
        xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'

        # If the .tpr file exists, run the gmx gyrate command
        if tpr_file.exists():
            # Construct the GROMACS gyrate command
            command = f'gmx gyrate -f {xtc_file} -s {tpr_file} -n {ndx_file} -o {gyrate_file}'
            user_input = '1'  # Select the appropriate group (typically group 1 is the molecule or complex)
            
            # Run the command to calculate the radius of gyration
            subprocess.run(command, shell=True, input=user_input, capture_output=True, text=True)
        else:
            continue

    return

def gyration_xvg_files_to_csvfiles(energyfolder_path, gyration_xvgfolder_path):
    ''' Go over all the xvg files in chronological order and create one big CSV file. 
        For .xvg files with multiple columns, only keep the first (picoseconds) and second columns.
    '''
    all_data = pd.DataFrame()  # Initialize an empty DataFrame
    
    for xvg_file in sorted(gyration_xvgfolder_path.glob('*.xvg')):   # Go over the xvg files in chronological order
        print(xvg_file)
        column_names = ['picoseconds', 'Gyration']  # Adjust column names for the first and second columns

        # Read the file and filter out lines starting with '#' or '@', while extracting legends
        with open(xvg_file, 'r') as file:
            lines = file.readlines()
            data_lines = []
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    continue  # Skip comment and metadata lines
                else:
                    data_lines.append(line)  # Collect data lines
        
        # Join the filtered lines into a single string
        data_str = ''.join(data_lines)  # This is the complete xvg file but without comments

        # Use StringIO to read the string as if it were a file
        data = pd.read_csv(StringIO(data_str), sep='\s+', header=None)

        # Select only the first (picoseconds) and second columns (SASA)
        data = data[[0, 1]]

        # Filter the data to match the time intervals you are interested in
        filtered_dataframe = data #data[data[0] % (public_variables.timeinterval_snapshots * 1000) == 0]

        # Rename the columns
        filtered_dataframe.columns = column_names

        # Add a column with the molecule ID (from the filename)
        filtered_dataframe.insert(0, 'mol_id', xvg_file.name.rsplit('.', 1)[0])

        # Append the selected columns to the all_data DataFrame
        all_data = pd.concat([all_data, filtered_dataframe], ignore_index=True)
        #print(all_data)

    # Save the concatenated data as a CSV file
    all_data.to_csv(energyfolder_path / 'gyration.csv', index=False)
    
    return

def make_index_files(MD_path):
    ''' function'''

    # Change the directory to MD_path once before the loop
    os.chdir(MD_path)
    user_input = 'q'

    for padded_num in (f"{i:03}" for i in range(1, 900)): #not abstract yet, not necessary
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
        os.chdir(MD_path / padded_num)
        print(tpr_file)
        if tpr_file.exists():
            # Construct the command and its arguments as a list
            command = ["gmx", "make_ndx", "-f", str(tpr_file), "-o", str(ndx_file)]
            
            # Run the command using subprocess and provide inputs interactively
            try:
                with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
                    # Provide the inputs to the command
                    proc.communicate(input='1 name ligand\nq\n')
                print(f"Index file created: {ndx_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running the command: {e}")
            except FileNotFoundError as e:
                print(f"Command not found: {e}")
        else:
            print(f'tpr not present in: {padded_num}')
            continue
    return

def calculate_hbond(MD_path):
    '''Function to compute hydrogen bonds using gmx hbond.'''

    # Change the directory to MD_path once before the loop
    os.chdir(MD_path)

    # Iterate through the padded range of numbers
    for padded_num in (f"{i:03}" for i in range(1, 857)):  # Adjust the range as needed
        # Define file paths
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'  # Trajectory file
        ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
        hbond_file = Path('/home/ben/Download/Afstuderen0/Afstuderen/energyfolder_files_JAK1/other_MD_feature_files/Hbond') / f'{padded_num}_hbond.xvg'
        print(type(tpr_file))
        print(type(hbond_file))
        
        print(tpr_file)
        print(xtc_file)
        print(ndx_file)
        print(hbond_file)
        
        if tpr_file.exists() and xtc_file.exists():
            # Construct the command and its arguments as a list
            command = ["gmx", "hbond", "-f", str(xtc_file), "-s", str(tpr_file), "-n", str(ndx_file), "-dist", str(hbond_file)]
            
            # Define the input strings for group selections
            # Assume group 2 is the donor and group 3 is the acceptor
            input_str = '2\n3\n'  # Adjust these numbers as needed
            
            # Run the command using subprocess and provide inputs interactively
            try:
                with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
                    # Provide the inputs to the command
                    proc.communicate(input=input_str)  # Send the input strings
                print(f"Hydrogen bond distances calculated: {hbond_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running the command: {e}")
            except FileNotFoundError as e:
                print(f"Command not found: {e}")
        else:
            print(f"Files not found: {tpr_file}, {xtc_file}")
            continue

    return

def calculate_hbond_dataframe_trajectory(MD_path, lig_conf_system_path):
    '''Function to compute hydrogen bonds using gmx hbond.'''

    # Change the directory to MD_path once before the loop
    os.chdir(MD_path)

    final_df = pd.DataFrame(columns=['mol_id', 'picoseconds', 'num_of_hbonds'])
    result_list = []
    # Iterate through the padded range of numbers
    for padded_num in (f"{i:03}" for i in range(1, 857)):  # Adjust the range as needed
        print(padded_num)
        # Define file paths
        os.chdir(MD_path / f'{padded_num}')
        xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
        hbnum_file = public_variables.MDsimulations_path_ / f'{padded_num}' / 'hbnum.xvg'
        hbond_file = public_variables.MDsimulations_path_ / f'{padded_num}' / f'{padded_num}_hbond_traj.xvg'
        if tpr_file.exists(): # and pdbfile_path.exists():
            #gmx hbond -f 001_prod.xtc -s 001_prod.tpr -n 001_index.ndx -dist 001_hbondtesting.xvg
            # Construct the command and its arguments as a list
            command = ["gmx", "hbond", "-f", str(xtc_file), "-s", str(tpr_file), "-n", str(ndx_file), "-dist", str(hbond_file)]
            
            # Define the input strings for group selections
            # Assume group 2 is the donor and group 3 is the acceptor
            input_str = '2\n3\n'  # Adjust these numbers as needed
            # Run the command using subprocess and provide inputs interactively
            try:
                print('try')
                with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
                    # Provide the inputs to the command
                    proc.communicate(input=input_str)  # Send the input strings
                    num_of_hbonds_df = read_out_hbnum(hbnum_file)
                    result_df = pd.DataFrame({
                        'mol_id': padded_num,
                        'picoseconds': num_of_hbonds_df.index*10,     # Original index
                        'num_of_hbonds': num_of_hbonds_df.values  # Values from num_of_hbonds
                    })
                    
                    print('append to result list')
                    result_list.append(result_df)
                    print('done appending')
            except subprocess.CalledProcessError as e:
                        print(f"Error occurred while running the command: {e}")
            except FileNotFoundError as e:
                print(f"Command not found: {e}")
        else:
            print(f"{padded_num} not valid")
            continue
    final_df = pd.concat(result_list, ignore_index=True)
    final_df.to_csv(public_variables.energyfolder_path_ / 'hbonds.csv', index=False)
    return final_df


def calculate_hbond_dataframe(MD_path, lig_conf_system_path):
    '''Function to compute hydrogen bonds using gmx hbond.'''

    # Change the directory to MD_path once before the loop
    os.chdir(MD_path)

    df = pd.DataFrame(columns=['mol_id', 'picoseconds', 'num_of_hbonds', 'average_hbond_distance'])

    # Iterate through the padded range of numbers
    for padded_num in (f"{i:03}" for i in range(1, 857)):  # Adjust the range as needed
        print(padded_num)
        # Define file paths
        os.chdir(MD_path / f'{padded_num}')
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
        hbnum_file = public_variables.MDsimulations_path_ / f'{padded_num}' / 'hbnum.xvg'
        hbond_file = public_variables.MDsimulations_path_ / f'{padded_num}' / f'{padded_num}_hbond.xvg'
        if tpr_file.exists(): # and pdbfile_path.exists():
            for ns in range(0,11):
                pdbfile_path = lig_conf_system_path / f'{ns}ns' / f'{padded_num}_{ns}ns.pdb'
                if pdbfile_path.exists():
                
                    # Construct the command and its arguments as a list
                    command = ["gmx", "hbond", "-f", str(pdbfile_path), "-s", str(tpr_file), "-n", str(ndx_file), "-dist", str(hbond_file)]
                    
                    # Define the input strings for group selections
                    # Assume group 2 is the donor and group 3 is the acceptor
                    input_str = '2\n3\n'  # Adjust these numbers as needed
                    
                    # Run the command using subprocess and provide inputs interactively
                    try:
                        print('try')
                        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
                            # Provide the inputs to the command
                            proc.communicate(input=input_str)  # Send the input strings
                        #print(f"Hydrogen bond distances calculated: {hbond_file}")
                        num_of_hbonds = read_out_hbnum(hbnum_file)
                        average_distance = read_out_hbond_xvgfile(hbond_file)
                        # Delete the files after processing
                        
                        os.remove(hbnum_file)
                        os.remove(hbond_file)

                        df.loc[len(df)] = [padded_num, ns*1000, num_of_hbonds, average_distance]

                    except subprocess.CalledProcessError as e:
                        print(f"Error occurred while running the command: {e}")
                    except FileNotFoundError as e:
                        print(f"Command not found: {e}")
                else:
                    print(f"Files not found: {tpr_file}, {pdbfile_path}")
                    continue
        else:
            print(f"{padded_num} not valid")
            continue
    df.to_csv(public_variables.energyfolder_path_ / 'hbonds.csv', index=False)
    return

def calculate_sasa(MD_path, lig_conf_system_path):
    '''Function to compute hydrogen bonds using gmx hbond.'''

    # Change the directory to MD_path once before the loop
    os.chdir(MD_path)

    df = pd.DataFrame(columns=['mol_id', 'picoseconds', 'num_of_hbonds', 'average_hbond_distance'])

    # Iterate through the padded range of numbers
    for padded_num in (f"{i:03}" for i in range(1, 857)):  # Adjust the range as needed
        print(padded_num)
        # Define file paths
        os.chdir(MD_path / f'{padded_num}')
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
        hbnum_file = public_variables.MDsimulations_path_ / f'{padded_num}' / 'hbnum.xvg'
        hbond_file = public_variables.MDsimulations_path_ / f'{padded_num}' / f'{padded_num}_hbond.xvg'
        if tpr_file.exists(): # and pdbfile_path.exists():
            for ns in range(0,11):
            
            
                pdbfile_path = lig_conf_system_path / f'{ns}ns' / f'{padded_num}_{ns}ns.pdb'
                if pdbfile_path.exists():
                
                    # Construct the command and its arguments as a list
                    command = ["gmx", "hbond", "-f", str(pdbfile_path), "-s", str(tpr_file), "-n", str(ndx_file), "-dist", str(hbond_file)]
                    
                    # Define the input strings for group selections
                    # Assume group 2 is the donor and group 3 is the acceptor
                    input_str = '2\n3\n'  # Adjust these numbers as needed
                    
                    # Run the command using subprocess and provide inputs interactively
                    try:
                        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
                            # Provide the inputs to the command
                            proc.communicate(input=input_str)  # Send the input strings
                        #print(f"Hydrogen bond distances calculated: {hbond_file}")
                        num_of_hbonds = read_out_hbnum(hbnum_file)
                        average_distance = read_out_hbond_xvgfile(hbond_file)
                        # Delete the files after processing
                        
                        os.remove(hbnum_file)
                        os.remove(hbond_file)

                        df.loc[len(df)] = [padded_num, ns*1000, num_of_hbonds, average_distance]

                    except subprocess.CalledProcessError as e:
                        print(f"Error occurred while running the command: {e}")
                    except FileNotFoundError as e:
                        print(f"Command not found: {e}")
                else:
                    print(f"Files not found: {tpr_file}, {pdbfile_path}")
                    continue
        else:
            print(f"{padded_num} not valid")
            continue
    df.to_csv(public_variables.energyfolder_path_ / 'hbonds.csv', index=False)
    return

def read_out_hbnum(xvg_file):
    df = pd.read_csv(xvg_file, sep="\s+", comment='@', header=None, skiprows=17)
    
    # Assuming the second column contains the values you're interested in
    # Extract the value at the row you're interested in; in this case, the value on the first row
    num_of_hbonds = df.iloc[:, 1] if not df.empty else None
    
    return num_of_hbonds

def read_out_hbond_xvgfile(xvg_file):
    df = pd.read_csv(xvg_file, sep="\s+", comment='@', header=None, skiprows=17)
        
    # Remove any remaining comment lines starting with '@'
    # Extract columns (assuming the first column is distance and the second column is frequency)

    distances = df[0]
    frequencies = df[1]
    
    # Calculate the weighted average distance
    total_weight = round(frequencies.sum())
    if total_weight == 0:
        return 0
    else:
        weighted_average_distance = (distances * frequencies).sum() / total_weight
    return weighted_average_distance

def run_gmx_sasa(MD_path, output_path):
    ''' create the 'xvg_files' folder to store the xvg files
        run the gmx energy command for every molecule
        !!! NEEDS TO BE RUN ON LINUX/BASH !!!
    '''
    output_path.mkdir(parents=True, exist_ok=True)
    # create the path for the xvg file to be stored
    for padded_num in (f"{i:03}" for i in range(1, 857)):
        print(padded_num)
        os.chdir(MD_path / f'{padded_num}')
        sasa_file = output_path / f'{padded_num}.xvg'
        xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
        # If the file already exists, modify the filename to avoid overwriting
        if tpr_file.exists():
        # Generate a new folder for the xvg files with a suffix to avoid overwriting
            command = f'gmx sasa -f {xtc_file} -s {tpr_file} -n {ndx_file} -o {sasa_file}'
            user_input = '2'
            #run the command which creates the xvg files
            subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)
        else:
            continue
    return

def sasa_xvg_files_to_csvfiles(energyfolder_path, sasa_xvgfolder_path):
    ''' go over all the xvg files in chronical order and create one big csv file which 
    '''
    all_data = pd.DataFrame()
    
    for xvg_file in sorted(sasa_xvgfolder_path.glob('*.xvg')):   #go over the xvg files/molecules in chronical order
        print(xvg_file)
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
        filtered_dataframe = data #data[data[0] % (public_variables.timeinterval_snapshots*1000) == 0]
        # Rename the columns with the extracted legends
        filtered_dataframe.columns = column_names
        # add a column with the mol id
        filtered_dataframe.insert(0, 'mol_id', xvg_file.name.rsplit('.', 1)[0])
        
        # Append the selected columns to the all_data DataFrame
        all_data = pd.concat([all_data, filtered_dataframe], ignore_index=True)
        print(all_data)
    # concatenate the dataframes under eachother and save them in a csv file
    #
    
    #MDfeatures_allmol_csvfile_ = generate_new_csv_filename(sasa_xvgfolder_path, public_variables.MDfeatures_allmol_csvfile)
    all_data.to_csv(energyfolder_path / 'sasa.csv', index=False)
    return

def run_gmx_rms(MD_path, output_path):
    ''' create the 'xvg_files' folder to store the xvg files
        run the gmx energy command for every molecule
        !!! NEEDS TO BE RUN ON LINUX/BASH !!!
    '''
    output_path.mkdir(parents=True, exist_ok=True)
    # create the path for the xvg file to be stored
    for padded_num in (f"{i:03}" for i in range(1, 857)):
        os.chdir(MD_path / f'{padded_num}')
        rmsd_file = output_path / f'{padded_num}.xvg'
        xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
        # If the file already exists, modify the filename to avoid overwriting
        if tpr_file.exists():
        # Generate a new folder for the xvg files with a suffix to avoid overwriting
            command = f'gmx rms -f {xtc_file} -s {tpr_file} -n {ndx_file} -o {rmsd_file}'
            user_input = '2\n2\n'
            #run the command which creates the xvg files
            subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)
        else:
            continue
    return

def rms_xvg_files_to_csvfiles(energyfolder_path, sasa_xvgfolder_path):
    ''' go over all the xvg files in chronical order and create one big csv file which 
    '''
    all_data = pd.DataFrame()
    
    for xvg_file in sorted(sasa_xvgfolder_path.glob('*.xvg')):   #go over the xvg files/molecules in chronical order
        print(xvg_file)
        column_names = ['picoseconds']
        # Read the file and filter out lines starting with '#' or '@', while extracting legends
        with open(xvg_file, 'r') as file:
            lines = file.readlines()
            data_lines = []
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    if line.startswith('@ s'):
                        # Extract the legend name
                        column_name = 'rmsd'
                        column_names.append(column_name)
                else:
                    data_lines.append(line)
        
        # Join the filtered lines into a single string
        data_str = ''.join(data_lines) #this is the complete xvg file but without the first 30 lines/comments or so

        # Use StringIO to read the string as if it were a file
        data = pd.read_csv(StringIO(data_str), sep='\s+', header=None)

        #NOTE: xvg file stays of equal length (independent of timeinterval_snapshot) so variable data as well
        #NOTE: variable 'filtered_dataframe' only contains the dataline at the time intervals we are interested in
        filtered_dataframe = data #data[data[0] % (public_variables.timeinterval_snapshots*1000) == 0]
        # Rename the columns with the extracted legends
        filtered_dataframe.columns = column_names
        # add a column with the mol id
        filtered_dataframe.insert(0, 'mol_id', xvg_file.name.rsplit('.', 1)[0])
        
        # Append the selected columns to the all_data DataFrame
        all_data = pd.concat([all_data, filtered_dataframe], ignore_index=True)

    # concatenate the dataframes under eachother and save them in a csv file
    #
    
    #MDfeatures_allmol_csvfile_ = generate_new_csv_filename(sasa_xvgfolder_path, public_variables.MDfeatures_allmol_csvfile)
    all_data.to_csv(energyfolder_path / 'rms.csv', index=False)
    return

def concatenate_csv_files(folder_path, csv_filenames):
    """
    Concatenate multiple CSV files into a single DataFrame.

    Parameters:
    - folder_path: str, the folder where the CSV files are located.
    - csv_filenames: list of str, the names of the CSV files to concatenate.

    Returns:
    - DataFrame: concatenated DataFrame containing data from all CSV files.
    """
    # List to store individual DataFrames
    dataframes = []

    # Loop through the list of CSV file names
    for i, filename in enumerate(csv_filenames):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Drop the first two columns
        if i == 0:
            df = df.iloc[:, :]
        else:
            df = df.iloc[:, 2:]
        # Append the DataFrame to the list
        dataframes.append(df)
    
    # Concatenate all DataFrames in the list horizontally (i.e., along columns)
    concatenated_df = pd.concat(dataframes, axis=1)
    
    return concatenated_df

def calculate_psa_from_trajectory(MD_path, output_file=None, interval_ns=1):
    # Load the topology and trajectory into an MDAnalysis Universe

    interval_ps = interval_ns * 1000

    sorted_files = sorted(MD_path.iterdir(), key=lambda x: int(x.stem))
    for molecule_path in sorted_files:
        xtc_files = list(molecule_path.glob("*.xtc"))
        if xtc_files: #check if valid MD simulation has run
            molecule_number = molecule_path.name

            topology_file = molecule_path / f'{molecule_number}_prod.tpr'
            trajectory_file = molecule_path / f'{molecule_number}_prod.xtc'
            
            if molecule_number == '021':
                l = []
                print('molecule')
                u = mda.Universe(topology_file, trajectory_file)
                # for res in u.residues:
                #     l.append(res.resname)
                polar_atoms = u.select_atoms('resid 1 and (name O* or name N*)')
                psa_results = []

                for ts in u.trajectory:
                    if ts.time % 1000 == 0:
                        print(mda.__version__)
                        # sasa_calculator = sasa.SASA(u, selection=f'resname {ligand_resname} and (name O or name N)')
                # print(polar_atoms)
                # for atom in u.select_atoms('resid 1'):
                #     print(atom.name)
                # print(l)
                # print(len(l))


                # # Initialize a list to store all frames and their corresponding times
                # frames = []
                # frame_times = []

                # # Iterate over all frames in the trajectory
                # for frame in u.trajectory:
                #     frames.append(frame)            # Store the frame (if needed)
                #     frame_times.append(frame.time)
                # print(frame_times)
    # # Convert the desired timestamps into MDAnalysis time units (usually picoseconds)
    # trajectory_times = np.array([ts for ts in u.trajectory.times])
    
    # # Initialize dictionary to store results
    # psa_results = {}

    # # Loop over desired timestamps
    # for timestamp in timestamps:
    #     # Find the frame closest to the desired timestamp
    #     closest_frame_idx = np.abs(trajectory_times - timestamp).argmin()
        
    #     # Set the trajectory to that frame
    #     u.trajectory[closest_frame_idx]
        
    #     # Calculate PSA
    #     psa_calculator = PSA.PSAnalysis(u)
    #     psa_calculator.run()

    #     # Store the PSA result
    #     psa_results[timestamp] = psa_calculator.results.psa
    #     print(f"PSA at {timestamp} ps: {psa_calculator.results.psa} Å²")

    # # Optionally save results to a file
    # if output_file:
    #     with open(output_file, 'w') as f:
    #         for timestamp, psa in psa_results.items():
    #             f.write(f"{timestamp}, {psa}\n")

    return #psa_results

def main():
    MDsimulations_path = public_variables.MDsimulations_path_
    
    #create ndx files
    #make_index_files(MDsimulations_path)
    lig_conf_system_path = public_variables.base_path_ / 'ligand_conformations_system'
    
    # calculate_hbond_dataframe(MD_path=MDsimulations_path, lig_conf_system_path = lig_conf_system_path)
    # pdb_file = Path('/home/ben/Download/Afstuderen0/MDsimulations/001/hbond_distances_try_pdb5.xvg')
    # print(read_out_hbond_xvgfile(pdb_file))
    # pdb_file = Path('/home/ben/Download/Afstuderen0/MDsimulations/001/hbnum.xvg')
    # print(read_out_hbnum(pdb_file))

    #calculate_hbond_dataframe(MD_path=MDsimulations_path, lig_conf_system_path = lig_conf_system_path)

    #use this for 
    # calculate_hbond_dataframe_trajectory(MD_path=MDsimulations_path, lig_conf_system_path = lig_conf_system_path)

    outputdir = public_variables.energyfolder_path_ / 'SASA'
    # run_gmx_sasa(MDsimulations_path, outputdir) #NOTE: done
    energyfolder_path = public_variables.energyfolder_path_
    # sasa_xvg_files_to_csvfiles(energyfolder_path, outputdir)
    outputdir = public_variables.energyfolder_path_ / 'RMSD'
    # run_gmx_rms(MDsimulations_path, outputdir)
    # rms_xvg_files_to_csvfiles(energyfolder_path, outputdir)

    #NOTE: gyration
    outputdir_gyration = public_variables.energyfolder_path_ / 'Gyration'
    run_gmx_gyrate(MD_path=MDsimulations_path, output_path=outputdir_gyration)
    gyration_xvg_files_to_csvfiles(energyfolder_path, outputdir_gyration)

    #NOTE: epsilon and total dipole moment
    # outputdir_TDM = public_variables.energyfolder_path_ / 'Total_dipoleMoment'
    # outputdir_epsilon = public_variables.energyfolder_path_ / 'epsilon'
    # run_gmx_dipoles(MD_path=MDsimulations_path, output_path_TDM=outputdir_TDM, output_path_epsilon=outputdir_epsilon)
    # Total_dipole_moment_xvg_files_to_csvfiles(energyfolder_path, totaldipoleMoment_xvgfolder_path=outputdir_TDM)
    # epsilon_xvg_files_to_csvfiles(energyfolder_path, epsilon_xvgfolder_path=outputdir_epsilon)

    outputdir_PSA = public_variables.energyfolder_path_ / 'PSA'
    #calculate_psa_from_trajectory(MDsimulations_path, output_file=outputdir_PSA) #dont use this i think
    # make_PSA_index_files(MDsimulations_path)
    # outputdir = public_variables.energyfolder_path_ / 'PSA'
    # run_gmx_psa_sasa(MDsimulations_path, outputdir)
    # psa_xvg_files_to_csvfiles(energyfolder_path, outputdir)

    # file_list = ['hbonds.csv', 'rms.csv', 'sasa.csv', 'psa.csv', 'epsilon.csv', 'gyration.csv', 'dipole_moment_total.csv',f'MD_features_{public_variables.dataset_protein_}.csv']
    # folder_path = public_variables.energyfolder_path_
    # dfs = concatenate_csv_files(folder_path, file_list)
    # dfs.to_csv(public_variables.energyfolder_path_ / 'MD_output.csv', index=False)
    return

main()