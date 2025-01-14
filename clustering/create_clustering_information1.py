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
# import MDAnalysis.analysis.psa

import numpy as np
from pathlib import Path

def order_MDsimulations_folders(MDsimulations_path):
    sorted_folders = sorted(
        [
            folder for folder in MDsimulations_path.iterdir()
            if folder.is_dir() and any(folder.glob("*prod.*"))
        ],
        key=lambda x: int(x.name)
    )
    return sorted_folders

def create_variables_cluster_files(molecule_number):
    cluster_files = {
        'sz': f"{molecule_number}_cluster_sizes.xvg",
        'g': f"{molecule_number}_cluster.log",
        'cl': f"{molecule_number}_central_structures.xtc", #gets the centroid (so smallest RMSD compared to all the others)
        'clid': f"{molecule_number}_cluster_ids.xvg",
        'dist': f"{molecule_number}_rmsd_dist.xvg",
        'o': f"{molecule_number}_rmsd_cluster_test.xpm",
        'clndx': f"{molecule_number}_cluster_indexing.ndx"
    }
    return cluster_files

def make_cluster_files(molecule_number, cluster_files, rmsd):
    # print(molecule_number)
    
    command = f"gmx cluster -f {molecule_number}_prod.xtc -s {molecule_number}_prod.tpr -n {molecule_number}_index.ndx -method gromos -cutoff {rmsd} -dist {cluster_files['dist']} -sz {cluster_files['sz']} -g {cluster_files['g']} -cl {cluster_files['cl']} -clid {cluster_files['clid']} -o {cluster_files['o']} -clndx {cluster_files['clndx']}"
    user_input = '\n'.join(["2","2"])
    subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)
    return

def get_clustering_information(cluster_log_file):
    #cluster_list and cluster_centroid_time_list are rebundant. all information is in the other two (indexing and /10)
    cluster_sizes = []
    cluster_centroid_frames = []
    
    with open(cluster_log_file, 'r') as file:
        for line in file:
            if not line.strip():  # This checks if the line is empty or only contains spaces
                continue
            parts = line.split()
            if parts[0] == 'Found':
                number_of_clusters = int(parts[1])
            elif parts[0].isdigit():
                if int(parts[0]) <= 10:
                    # Only process lines where parts[0] is a digit and less than 10
                    cluster_size = int(parts[2])
                    cluster_centroid_time = int(parts[5] if cluster_size > 1 else parts[4])  # clusters with size 1 are different
                    cluster_sizes.append(cluster_size)
                    cluster_centroid_frames.append(cluster_centroid_time // 10)
                else:
                    break
    # Construct the clustering_information list for clarity
    clustering_information = [number_of_clusters, cluster_sizes, cluster_centroid_frames]
    return clustering_information


def read_out_cluster_files(cluster_files):
    #sz is cluster sizes
    cluster_sizes = get_cluster_sizes(cluster_files['sz'])

    # g contains average rmsd etc
    get_clustering_information(cluster_files['g'])
    return cluster_sizes

def get_mean_rmsd_value(rmsd_distribution_xvgfile):
    mean_rmsd = 0
    data = np.loadtxt(rmsd_distribution_xvgfile, comments=['#', '@'])
    rmsd_values = data[:, 0]  # RMSD values
    frequencies = data[:, 1]  # Frequency of each RMSD
    print(rmsd_values)
    print(frequencies)
    weighted_mean = (rmsd_values * frequencies).sum() / frequencies.sum()
    print(weighted_mean)
    print(rmsd_values.mean())
    return mean_rmsd

def get_cluster_sizes(cluster_sizes_xvgfile):
    with open(cluster_sizes_xvgfile, 'r') as file:
        lines = file.readlines()
        cluster_sizes = []
        for line in lines:
            if line.startswith('#') or line.startswith('@'):
                continue
            else:
                parts = line.split()
                cluster_size = parts[1]
                cluster_sizes.append(int(cluster_size))
    return cluster_sizes

def remove_cluster_files(cluster_files):
    # You can now use the filenames stored in the dictionary to remove the files
    for file in cluster_files.values():
        if Path(file).exists():
            Path(file).unlink()  # Remove the file
    return

def get_centroid_from_log_file(logfile):
    with open(logfile, 'r') as file:
        for line in file:
            # Check if the line starts with "1" (the first cluster)
            if line.startswith("  1"):
                # Split the line into parts and extract the middle frame (usually 5th value)
                parts = line.split()
                centroid_timepoint = parts[5]  # This is where the middle frame number appears in your log output
                centroid_frame = int(int(centroid_timepoint) / 10)
    return centroid_timepoint, centroid_frame

def create_pdb_file_of_centroid(molecule_number, centroid_timepoint):
    pdb_file = f'{molecule_number}_centroid.pdb'
    command = f"gmx trjconv -s {molecule_number}_prod.tpr -f {molecule_number}_prod.xtc -o {pdb_file} -b {centroid_timepoint} -e {centroid_timepoint}"
    user_input = '\n'.join(["2"])
    subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)
    #remove lp
    with open(pdb_file, 'r') as infile, open(str(pdb_file) + '_tmp', 'w') as tmpfile:
        for line in infile:
            if ' LP' not in line:
                tmpfile.write(line)
    shutil.move(str(pdb_file) + '_tmp', pdb_file)
    full_path = os.path.abspath(pdb_file)
    return full_path

def create_folder_stable_conformations_method_one(full_path_pdb_file):
    base_path = public_variables.base_path_
    destination_path = base_path / 'stable_conformations_method_one'
    destination_path.mkdir(parents=True, exist_ok=True)
    dst_filename = os.path.join(destination_path, os.path.basename(full_path_pdb_file)) #ensure we prevent errors
    shutil.move(full_path_pdb_file, dst_filename)
    return

def create_pdb_with_one_rmsd(smiles_strings, sorted_folders, goal_amount_of_clusters=20):
    
    for molecule_folder in sorted_folders: #NOTE: remove [1:2], [107:108] is 125 with LP lonepairs.
        os.chdir(molecule_folder)
        
        #get RDkit features
        molecule_number = molecule_folder.name
        print(molecule_number)
        mol_id = int(molecule_number)
        mol = Chem.MolFromSmiles(smiles_strings[mol_id])
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        print(f"rotatable bonds: {num_rotatable_bonds}")
        goal_amount_of_clusters = 10+2*int(num_rotatable_bonds)
        heavy_atoms = mol.GetNumHeavyAtoms()
        dic = {}
        if num_rotatable_bonds < 6:
            rmsd_range = [0.025,0.035,0.05,0.075,0.1]
        else:
            rmsd_range = [0.1,0.15,0.2,0.25] #0.025 too small
        cluster_files = create_variables_cluster_files(molecule_number) #all file names, xvg xmp ndx etc
         #try to remove the files so we dont run out of space
        remove_cluster_files(cluster_files)
        for rmsd in rmsd_range:
            
            make_cluster_files(molecule_number, cluster_files, rmsd)
            cluster_sizes = read_out_cluster_files(cluster_files)
            dic[f'rmsd{rmsd}'] = cluster_sizes
            print(len(cluster_sizes))
            remove_cluster_files(cluster_files)
        closest_key = min(dic, key=lambda k: abs(len(dic[k]) - goal_amount_of_clusters))
        rmsd_value = float(closest_key.replace('rmsd', ''))
        print(f'best rmsd value: {rmsd_value}')
        make_cluster_files(molecule_number, cluster_files, rmsd_value)

        #get centroid now
        centroid_timepoint, centroid_frame = get_centroid_from_log_file(cluster_files['g'])

        #get pdb file of the centroid of cluster 1
        full_path_pdb_file = create_pdb_file_of_centroid(molecule_number, centroid_timepoint)
        create_folder_stable_conformations_method_one(full_path_pdb_file)
    return

def clustering(smiles_strings, valid_sorted_folders):
    all_clustering_information = []

    # Check if the file exists
    file_path = public_variables.base_path_ / f'clustering_information_{public_variables.dataset_protein_}.csv'
    if os.path.exists(file_path):
        # If it exists, read it into a DataFrame
        df_existing = pd.read_csv(file_path)
    else:
        # If the file doesn't exist, create an empty DataFrame with the same columns
        columns = ['mol_id', 'RMSD', 'Number of clusters', 'Cluster Sizes', 'Cluster Centroids', 'Number of Rotatable Bonds']
        df_existing = pd.DataFrame(columns=columns)

    for molecule_folder in valid_sorted_folders[340:350]: #NOTE: remove [1:2], [107:108] is 125 with LP lonepairs.
        os.chdir(molecule_folder)
        molecule_number = molecule_folder.name
        print(molecule_number)

        mol_id = int(molecule_number)
        mol = Chem.MolFromSmiles(smiles_strings[mol_id]) #smiles are indexed at mol_id
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        cluster_files = create_variables_cluster_files(molecule_number)
        remove_cluster_files(cluster_files) #remove old cluster files

        rmsd = 0.01
        step_size = 0.005

        #do this to reduce time step in the beginning
        # Loop to adjust RMSD
        while rmsd <= 0.35:
            
            make_cluster_files(molecule_number, cluster_files, rmsd)
            clustering_information = get_clustering_information(cluster_files['g']) #get all information from the log file
            remove_cluster_files(cluster_files)
            
            #insert the molecule number and rmsd to keep track
            clustering_information.insert(0, molecule_number)
            clustering_information.insert(1, rmsd)
            clustering_information.append(num_rotatable_bonds)
            print(clustering_information)
            # Check number of clusters and adjust step size
            size_cluster_one = clustering_information[3][0]  # Assuming clustering_information[2] holds cluster sizes
            if size_cluster_one < 30:
                step_size = 0.01  # If cluster 1 is bigger than 50 decrease step size
            else:
                step_size = 0.005  # If number of clusters is less than 500, revert to the smaller step size
            rmsd = np.round(rmsd + step_size, 4)
            
            
            #add some benchmarks to shorten the code and reduce the size of the dataframe
            if clustering_information[3][0] < 90: #if first cluster is smaller than 90 molecules, continue
                continue
            #if first cluster has more than 700 molecules or less then 10 clusters, break, reduces time
            if clustering_information[3][0] > 600:# or clustering_information[2] < 5: 
                break
            all_clustering_information.append(clustering_information)
            
            
            
    # Convert new data to a DataFrame
    new_data = pd.DataFrame(all_clustering_information, columns=['mol_id', 'RMSD', 'Number of clusters', 'Cluster Sizes', 'Cluster Centroids', 'Number of Rotatable Bonds'])
    
    # Append new data to the existing DataFrame
    df_combined = pd.concat([df_existing, new_data], ignore_index=True)
    
    # Save the combined DataFrame back to the CSV
    df_combined.to_csv(file_path, index=False)

    #df = pd.DataFrame([data], columns=['ID', 'RMSD', 'Cluster Sizes', 'Cluster Centroids'])
    #if cluster group is more than 50%, stop the clustering. or 80%. idk. 

    #needs to get into dataframe containing:
    #molecule, cluster with 50%, 40%, 30%, 20%
    #also all cluster information with smaller and smaller rmsd
    #so some kind of log file

    #dictionary, 001, rmsd 0.025, [cluster 1 size, cluster 2 size, cluster 3 size], [centroid cluster 1, centroid cluster 2]
    #               , rmsd 0.05 , [cluster 1 size, cluster 2 size, cluster 3 size], [centroid cluster 1, centroid cluster 2]
    # do a dataframe. we want it to csv? yes i think so. log file too much work.
    # so run it all and read it all out, and then build the dataframe. build it with lists
    #use
    #get_cluster_sizes(cluster_sizes_xvgfile)
    #make_cluster_files(molecule_number, cluster_files, rmsd)
    #create_variables_cluster_files(molecule_number)
    return

def main():
    MDsimulations_path = public_variables.MDsimulations_path_
    dataset_path = public_variables.dataset_path_
    
    dataset_dataframe = pd.read_csv(dataset_path)
    dataset_dataframe.set_index("mol_id", inplace=True)
    smiles_strings = dataset_dataframe.loc[:,"smiles"] #series

    valid_sorted_folders = order_MDsimulations_folders(MDsimulations_path) #all valid folders sorted (so valid molecules)
    
    clustering(smiles_strings, valid_sorted_folders)
    #NOTE: something is redundant
    # dic = {}
    # create_pdb_with_one_rmsd(smiles_strings, valid_sorted_folders, goal_amount_of_clusters=20)

    # #go 1 by 1 over the molecules
    # for molecule_folder in valid_sorted_folders[0:2]: #NOTE: remove [1:2], [107:108] is 125 with LP lonepairs.
    #     os.chdir(molecule_folder)
    #     molecule_number = molecule_folder.name
    #     logfile = f"{molecule_number}_cluster.log"
    #     centroid_timepoint, centroid_frame = get_centroid_from_log_file(logfile)
    #     # create_pdb_with_one_rmsd(smiles_strings, sorted_folders, goal_amount_of_clusters=20)

        # #NOTE: to try to get better pdb files, was not necessary tho
        # cluster_files = create_variables_cluster_files(molecule_number)
        # #read out log file
        # centroid_timepoint, centroid_frame = get_centroid_from_log_file(cluster_files['g'])

        # #get pdb file from single frame
        # tprfile = f"{molecule_number}_prod.tpr" #TODO: maybe get whole path
        # xtcfile = f"{molecule_number}_prod.xtc"

        # u = mda.Universe(tprfile, xtcfile)
        # residue_1 = u.select_atoms("resid 1")
        # print(centroid_frame)
        # u.trajectory[centroid_frame]
        # output_pdb_file = f'{molecule_number}_centroid_{centroid_frame}.pdb'
        # with PDB.PDBWriter(output_pdb_file) as pdb:
        #         pdb.write(residue_1)
        # with open(output_pdb_file, 'r') as infile, open(str(output_pdb_file) + '_tmp', 'w') as tmpfile:
        #         for line in infile:
        #             if ' LP' not in line:
        #                 tmpfile.write(line)
        # shutil.move(str(output_pdb_file) + '_tmp', output_pdb_file)
        # destination_path = public_variables.base_path_ / 'stable_conformations_method_one'
        # destination_path.mkdir(parents=True, exist_ok=True)
        # os.path.abspath(output_pdb_file)
        # dst_filename = os.path.join(destination_path, os.path.basename(output_pdb_file)) #ensure we prevent errors
        # shutil.move(output_pdb_file, dst_filename)

if __name__ == "__main__":
    main()