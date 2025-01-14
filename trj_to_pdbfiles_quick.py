#use the 'simulations' folder and count how many molecules there are
#then create the 'ligandconformations' folder with subfolders 0ns to 10ns.
#put in every subfolder, the frame corresponding to the timestep of every molecule where the simulation ran.

import os
import numpy as np
from pathlib import Path
import pathlib
import subprocess
import re
import MDAnalysis as mda
from MDAnalysis.coordinates import PDB
import shutil
from global_files import public_variables

def get_molecules_lists(MDsimulations_path):
    '''uses the MD_simulations folder and checks for every molecule the simulation whether it contains the .tpr and .xtc file
        to see if it contains a valid trajectory file (.tpr)
        output: lists of all the molecules, valid molecules, and invalid molecules in strings
        '''
    molecules_list = []
    valid_mols = []
    invalid_mols = []
    print('Get Molecules (all/valids/invalids)')
    for item in MDsimulations_path.iterdir(): #item is every folder '001' etc in the MD folder
        
        tprfile = f"{item.name}_prod.tpr"
        xtcfile = f"{item.name}_prod.xtc"  # trajectory file
        
        trajectory_file = item / xtcfile

        if item.is_dir():  # Check if the item is a directory
            molecules_list.append(item.name)
        if not trajectory_file.exists():
            invalid_mols.append(item.name)
        else:
            valid_mols.append(item.name)

    return molecules_list, valid_mols, invalid_mols #['001','002']

#also creates the folder "ligandconformations/0ns" etc
def trj_to_pdb(valid_molecules_list, frames_to_extract,base_path, MDsimulations_path, output_foldername):
    ''' 
    
    '''
    print(MDsimulations_path)
    # Create directories once for all frames ()
    #TODO: this can be done in the forloop of forloop is we say x= 0 and then do it
    for f in frames_to_extract:
        ns_foldername = f / 100
        if ns_foldername.is_integer():
            ns_foldername = int(ns_foldername)
        dir = Path(f'{output_foldername}/{ns_foldername}ns')
        final_path = base_path / dir
        
        # Check if final_path exists and is a file, remove it if so
        if final_path.exists() and final_path.is_file():
            print('###################### this is final path and file')
            print(final_path)
            final_path.unlink()  # Remove the file
    
        # Create the directory
        final_path.mkdir(parents=True, exist_ok=True)


    #go over all the valid molecules and get the right snapshot of the trajectory into the right folders of 'ligand_conformations_for_every_ns'
    for mol in valid_molecules_list:
        print(mol)
        mol_path = MDsimulations_path / mol
        print(mol_path)
        if mol_path.exists() and mol_path.is_dir():
            os.chdir(MDsimulations_path / mol)
        print(output_foldername)
        topology_file = f'{mol}_prod.tpr'
        trajectory_file = f'{mol}_prod.xtc'
        
        # print(f'{public_variables.base_path_ / output_foldername / frame}.pdb')
        outputfolder = public_variables.base_path_ / output_foldername / mol
        command = f"gmx trjconv -s {topology_file} -f {trajectory_file} -o {outputfolder}_.pdb -sep -skip 10 -conect"
        print(command)
        # specify the user input which are the features
        user_input = '\n'.join(["2"])

        #run the command which creates the xvg files
        subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)

        #TODO: remove LP + process the name
        os.chdir(public_variables.base_path_ / output_foldername)
        print(os.getcwd())
        time_step_ns = 0.1
        print(list(sorted(os.listdir())))
        for filename in sorted(os.listdir()):
            print('filename')
            print(filename)
            if filename.startswith(f"{mol}_") and filename.endswith(".pdb"):
                x = re.search(f"{mol}_(\d+)\.pdb", filename).group(1)
                print('x')
                print(x)
                number = float(re.search(f"{mol}_(\d+)\.pdb", filename).group(1))
                print(number)
                timepoint = round(time_step_ns * number, 1)
                if timepoint.is_integer():
                    timepoint = int(timepoint)
                print('timepoint')
                print(timepoint)
                
                new_name = f"{mol}_{timepoint}ns.pdb"
                destination = public_variables.base_path_ / output_foldername / f'{timepoint}ns'
                print(destination)
                # Move the file first
                shutil.move(f'{public_variables.base_path_ / output_foldername / filename}', destination)

                # Then rename the file after moving
                os.rename(f'{destination}/{filename}', f'{destination}/{new_name}')
                with open(f'{destination}/{new_name}', 'r') as infile, open(str(f'{destination}/{new_name}') + '_tmp', 'w') as tmpfile:
                    for line in infile:
                        if ' LP' not in line:
                            tmpfile.write(line)
                # Replace the original file with the temporary file
                shutil.move(str(f'{destination}/{new_name}') + '_tmp', f'{destination}/{new_name}')
                # print(MDsimulations_path / mol / topology_file)
                # u = mda.Universe(MDsimulations_path / mol / topology_file, f'{destination}/{new_name}')
                # print(u)
                # residue_1 = u.select_atoms("resid 1")
                # print(residue_1)
                # with PDB.PDBWriter(f'{destination}/000000.pdb') as pdb_writer:
                #     # Write only the ligand atoms but using the full topology for CONECT records
                #     print('WRITNG something')
                #     pdb_writer.write(residue_1)
                # u = mda.Universe(MDsimulations_path / mol / topology_file,f'{destination}/{new_name}')
                # residue_1 = u.select_atoms("resid 1")
                # print(residue_1)
                # with PDB.PDBWriter(f'{destination}/0000.pdb') as pdb_writer:
                #     pdb_writer.write(residue_1)



        
    return

#NOTE: uses the MD simulations folder to create the folder 'ligand_conformations_for_every_snapshot'
def main(MDsimulations_path = public_variables.MDsimulations_path_, output_folder = public_variables.ligand_conformations_folder_):
    base_path = public_variables.base_path_
    
    # Create molecules list. so all molecules that are present in the folder 'simulations' get added to the list
    all_molecules_list, valid_mols, invalid_mols = get_molecules_lists(MDsimulations_path)
    print(all_molecules_list)
    print(valid_mols)
    print(invalid_mols)
    print('done getting molecules')
    #Define frames to extract
    frames_to_extract = list(range(0,1001,10)) #101 frames
    valid_mols_sorted = sorted(valid_mols, key=int)
    
    # Process trajectories
    trj_to_pdb(valid_mols_sorted[0:], frames_to_extract,base_path, MDsimulations_path, output_folder) #at 748
    print(f"number of molecules: {len(all_molecules_list)}")
    print(f"number of molecules with succesful simulations: {len(valid_mols)}")
    print(f"Invalid molecules: {invalid_mols}")
    return

# Example usage
if __name__ == "__main__":
    #NOTE: 10 seconden per molecule als ik 100 frames doe
    print('trj_to_pdbfiles')
    MDsimulations_path = public_variables.MDsimulations_path_ #the folder containing all the MD simulations {001,002..615}
    
    output_folder = public_variables.ligand_conformations_path_
    # ligand_conformations_folder_ = f'ligand_conformations_JAK1_2'
    ligand_conformations_folder_ = f'ligand_conformations_pparD'
    
    print(type(MDsimulations_path))
    main(MDsimulations_path, ligand_conformations_folder_)