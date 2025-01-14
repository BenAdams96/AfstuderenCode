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


# Function to extract the closest row for each molecule
def get_closest_rows(df, target=500):
    df['First Cluster Size'] = df['Cluster Sizes'].apply(lambda x: ast.literal_eval(x)[0])
    closest_rows = df.loc[df.groupby('mol_id')['First Cluster Size'].apply(lambda group: (group - target).abs().idxmin())]
    df = df.drop(columns=['First Cluster Size'])  # Optional: Remove the helper column
    return closest_rows

def create_pdb_files_for_cluster(molecule_number, centroid_timepoint):
    os.chdir(public_variables.MDsimulations_path_ / molecule_number)
    pdb_file = f'{molecule_number}_{centroid_timepoint}.pdb'
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
    return  full_path

def main():
    file_path = public_variables.base_path_ / 'code' / 'clustering' /  f'clustering_information_{public_variables.dataset_protein_}.csv'
    clustering_folder = public_variables.base_path_ / 'clustering folder' / public_variables.dataset_protein_
    clustering_folder.mkdir(parents=True, exist_ok=True)
    #50% 40$ 30% 20% 10% cluster 1 of total clusters
    #for each, cluster 1 2 3 5 and 8
    
    clustering_df = pd.read_csv(file_path)
    targets = [500,400,300,200,100]
    cluster_folders = [1,2,3,4,5,6,7,8,9,10]
    for target in targets:
        closest_rows = get_closest_rows(clustering_df, target=target)
        closest_rows['Cluster Centroids'] = closest_rows['Cluster Centroids'].apply(ast.literal_eval)
        print(target)
        #create the 5 pdb folders from this, for each cluster

        for cluster in cluster_folders:
            print(cluster)
            cluster_centroids = closest_rows['Cluster Centroids'].apply(
                lambda x: x[cluster - 1] if len(x) >= cluster else None
            )
            cluster_centroids.index = closest_rows['mol_id']
            if cluster_centroids.isnull().any():
                break
            #create folder for it
            destination_path = clustering_folder / f'clustering_target{target/10}%_cluster{cluster}'
            destination_path.mkdir(parents=True, exist_ok=True)

            #put all pdb files in it (include frame with the name)
            #so loop over the molecule numbers
            for molecule, centroid_frame in cluster_centroids.items():
                fwpadding_molecule = f"{molecule:03}"
                centroid_timepoint = centroid_frame*10
                full_pdb_file_path = create_pdb_files_for_cluster(fwpadding_molecule, centroid_timepoint)
                dst_filename = os.path.join(destination_path, os.path.basename(full_pdb_file_path)) #ensure we prevent errors
                shutil.move(full_pdb_file_path, dst_filename)


        # print(closest_rows['Cluster Centroids'][6])
        # full_path_pdb_file = create_pdb_files_for_cluster(closest_rows.iloc['Cluster Centroids'])
        

    # MDsimulations_path = public_variables.MDsimulations_path_
    # dataset_path = public_variables.dataset_path_
    
    # dataset_dataframe = pd.read_csv(dataset_path)
    # dataset_dataframe.set_index("mol_id", inplace=True)
    # smiles_strings = dataset_dataframe.loc[:,"smiles"] #series

    # valid_sorted_folders = order_MDsimulations_folders(MDsimulations_path) #all valid folders sorted (so valid molecules)

    # clustering(smiles_strings, valid_sorted_folders)
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