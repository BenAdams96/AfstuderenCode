import randomForest_read_in_models
from randomForest_Class import RandomForestModel
from randomForest_make_originalModels import Kfold_Cross_Validation_new
from randomForest_make_originalModels import visualize_scores_box_plot
import csv_to_dictionary
from sklearn.preprocessing import StandardScaler
from csv_to_dataframes import csvfiles_to_dfs

import public_variables
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
import numpy as np
from pathlib import Path
import shutil
from itertools import product

import pandas as pd
import math
import re
import os

def extract_k_and_scoring(filename):
    # Use regex to find the pattern 'k' followed by digits and the scoring method
    match = re.search(r'k(\d+)_(.+)\.pkl', filename)
    if match:
        k_value = int(match.group(1))
        scoring_metric = match.group(2)
        return k_value, scoring_metric
    else:
        return None, None
    
def get_molecules_lists_temp(parent_path):
    folder = public_variables.dfs_descriptors_only_path_
    csv_file = '0ns.csv'
    final_path = parent_path / folder / csv_file
    molecules_list = []
    invalid_mols = []
    df = pd.read_csv(final_path)
    mol_id_column = df['mol_id']

    valid_mol_list_str = list(map(str, mol_id_column))
    print(valid_mol_list_str)
    print(len(valid_mol_list_str))
    return  valid_mol_list_str

def extract_number(filename):
    return int(filename.split('ns.csv')[0])

def copy_redfolder_only_csv_files(source_folder, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            # Copy each CSV file to the destination folder
            shutil.copy2(source_file, destination_file)

def concatenate_group(group):
    # Separate out the PKI and mol_id
    mol_id = group['mol_id'].iloc[0]
    pki_value = group['PKI'].iloc[0]
    
    # Drop 'PKI' and 'mol_id' from the group to avoid duplication in concatenation
    group = group.drop(columns=['mol_id', 'PKI'])
    
    # Concatenate the remaining columns horizontally (axis=1)
    concatenated = pd.concat([group.reset_index(drop=True)], axis=1).T.reset_index(drop=True)
    
    # Flatten the column names
    concatenated.columns = [f"{col}_{i}" for i, col in enumerate(concatenated.columns)]
    
    # Add the 'mol_id' and 'PKI' back to the start
    concatenated.insert(0, 'mol_id', mol_id)
    concatenated.insert(1, 'PKI', pki_value)
    
    return concatenated

def MD_features_implementation(base_path):
    reduced_dataframes_folder = base_path / public_variables.dfs_reduced_path_
    destination_folder = base_path / public_variables.dfs_reduced_and_MD_path_
    csv_MDfeatures_file = public_variables.energyfolder_path_ / 'MD_output.csv' #csv file with all the succesfull molecules and their MD simulation features for every ns
    
    df_MDfeatures = pd.read_csv(csv_MDfeatures_file)
    df_MDfeatures['picoseconds'] = df_MDfeatures['picoseconds'] / 1000
    # remove the folder with added MD if it already exists
    if destination_folder.exists():
        if destination_folder.is_dir():
            # Remove the existing destination folder
            shutil.rmtree(destination_folder)

    #copy_redfolder_only_csv_files(reduced_dataframes_folder, destination_folder)
    os.makedirs(destination_folder, exist_ok=True)
    shutil.copy(csv_MDfeatures_file, destination_folder)
    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic(reduced_dataframes_folder, exclude_files=['concat_ver.csv', 'concat_hor.csv','rdkit_min.csv','MD_output.csv', 'conformations_1000.csv']) # , '0ns.csv', '1ns.csv', '2ns.csv', '3ns.csv', '4ns.csv', '5ns.csv', '6ns.csv', '7ns.csv', '8ns.csv', '9ns.csv', '10ns.csv'
    dfs_in_dic_concat = csv_to_dictionary.csvfiles_to_dic(reduced_dataframes_folder, exclude_files=['rdkit_min', '0ns', '1ns', '2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '8ns', '9ns', '10ns'])
    #dfs_in_dic = csv_to_dictionary.csvfiles_to_dic(reduced_dataframes_folder, exclude_files=[public_variables.MDfeatures_allmol_csvfile])

    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys()))
    dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic}
    print(dfs_in_dic.keys())
    #csv_files = sorted([f for f in os.listdir(reduced_dataframes_folder) if f.endswith('.csv')], key=extract_number)
    dfs_in_dic_MD = {}
    offset = 0 #if there are any useless

    for name, df in list(dfs_in_dic.items()):
        if name.startswith('conformations'):
            print(name)
            merged_df = pd.merge(df, df_MDfeatures, left_on=['mol_id', 'conformations (ns)'], right_on=['mol_id', 'picoseconds'], how='left')
            merged_df = merged_df.drop(columns='picoseconds')
            merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
            print(f'done with {name}')
            # offset = int(float(name.removesuffix('ns'))*(1/public_variables.timeinterval_snapshots))
            # print(offset)
            # MD_df = df_MDfeatures.iloc[offset::(int(10/public_variables.timeinterval_snapshots)+1)] #needs to be 11 and 21, not 22!!!
            # MD_df = MD_df.reset_index(drop=True)
            # combined_df = pd.concat([df, MD_df.iloc[:, 2:]], axis=1) #2: because we dont want mol_id and picoseconds as features
            # combined_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
        elif name.endswith('ns'):
            print(name)
            df_MDfeatures2 = df_MDfeatures[df_MDfeatures['picoseconds'] == int(name.rstrip('ns'))]
            print(df)
            merged_df = pd.merge(df, df_MDfeatures2, on='mol_id', how='left')
            merged_df = merged_df.drop(columns='picoseconds')
            merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
            print(f'done with {name}')
            
        elif name == 'concat_hor':
            df_MDfeatures_temp = df_MDfeatures[df_MDfeatures['picoseconds'] != 0].reset_index(drop=True) #we dont want 0ns in the dataframe
            # Drop 'mol_id' and 'picoseconds' from df_MDfeatures_temp
            df_MDfeatures_temp = df_MDfeatures_temp.drop(columns=['mol_id', 'picoseconds'])
            combined_df = pd.concat([dfs_in_dic['concat_ver'], df_MDfeatures_temp], axis=1) #add the two dataframes together
            combined_df['row_id'] = combined_df.groupby('mol_id').cumcount()

            # Extract the unique mol_id and PKI for each molecule
            pki_df = combined_df[['mol_id', 'PKI']].drop_duplicates(subset='mol_id').reset_index(drop=True)
            
            # Drop the PKI column from the original dataframe before concatenation
            combined_df_without_pki = combined_df.drop(columns=['PKI'])

            # Create a row identifier within each mol_id group for the 9 conformations
            combined_df_without_pki['row_id'] = combined_df_without_pki.groupby('mol_id').cumcount()

            # Pivot the dataframe to concatenate rows with the same mol_id
            df_wide = combined_df_without_pki.pivot(index='mol_id', columns='row_id')

            # Flatten the MultiIndex in the columns
            df_wide.columns = [f'{col[0]}_{col[1]}' for col in df_wide.columns]

            # Reset the index so mol_id is part of the dataframe
            df_wide = df_wide.reset_index()

            # Add the PKI column back as the second column in the final dataframe
            df_final = pd.merge(pki_df[['mol_id', 'PKI']], df_wide, on='mol_id')

            # Reorder columns to make PKI the second column
            columns_order = ['mol_id', 'PKI'] + [col for col in df_final.columns if col not in ['mol_id', 'PKI']]
            df_final = df_final[columns_order]
            
            df_final.to_csv(destination_folder / Path(name + '.csv'), index=False)
        else:
            df.to_csv(destination_folder / Path(name + '.csv'), index=False)
            continue

    # for ns_csvfile in csv_files:
    #     print(offset)
    #     df = pd.read_csv(reduced_dataframes_folder / ns_csvfile)
    #     temp_df = df_MDfeatures.iloc[offset::11] #why 11? because there are 11 snapshots (10ns including 0ns so in total 11ns)
    #     temp_df.reset_index(drop=True, inplace=True)

    #     df = pd.concat([df, temp_df.iloc[:, 2:]], axis=1) #leave out mol_id and picoseconds which are in the MD features csv file
    #     df.to_csv(destination_folder / ns_csvfile, index=False)
    #     print(list(df.columns))
    #     #needs to be dfs and not 1 df
    #     red_MD_dfs.append(df)
    #     offset+=1
    # parameter_grid = public_variables.parameter_grid_
    # modelresults = destination_folder / public_variables.Modelresults_
    # for kfold, (scoring, scoring_metric) in product(parameter_grid['kfold_'], parameter_grid['scoring_']):
    #     reduced_models, df_results = Kfold_Cross_Validation_new(red_MD_dfs, public_variables.hyperparameter_grid_, kfold_=kfold, scoring_=scoring)
    #     csv_filename = f'results_K{kfold}_{scoring_metric}.csv'
    #     os.makedirs(modelresults, exist_ok=True)
    #     visualize_scores_box_plot(df_results, kfold,scoring_metric, modelresults)
    #     df_results.to_csv(modelresults / csv_filename, index=False)
    # csv_files = sorted([f for f in os.listdir(destination_folder) if f.endswith('.csv') and not f.startswith('MD_features')], key=extract_number)
    # offset = 0 #if there are any useless
    # for nscsvfile in csv_files:
    #     print(offset)
    #     df = pd.read_csv(destination_folder / nscsvfile)
    #     temp_df = df_MDfeatures.iloc[offset::11]
    #     temp_df.reset_index(drop=True, inplace=True)

    #     df = pd.concat([df, temp_df], axis=1)
    #     df.to_csv(path / nscsvfile, index=False)
    #     print(list(df.columns))
    #     #needs to be dfs and not 1 df
    #     dfs.append(df)

    # for outer_key in dic_models:
    #     kandscoring = extract_k_and_scoring(outer_key)
    #     kfold_value = kandscoring[0]
    #     if kandscoring[1] == 'RMSE':
    #         scoring_metric = 'neg_root_mean_squared_error'
    #     elif kandscoring[1] == 'R-squared (R²)':
    #         scoring_metric = 'r2'
    #     print('outerkey')
    #     print(outer_key)
        


        # for inner_key in dic_models[outer_key]:
        #     dfs = [] #NOTE: needs to be somewhere else
        #     print(inner_key)
        #     if inner_key != 'original models': #there are no folders with 'original models'
        #         path = base_path / destination_folder / outer_key.replace('RF_Allmodels_', '').replace('.pkl', '') / inner_key
        #     elif inner_key == 'original models':
        #         path = base_path / destination_folder / outer_key.replace('RF_Allmodels_', '').replace('.pkl', '') / inner_key
        #         print(path)
        #         continue
        #     csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv') and f != 'results.csv' ], key=extract_number)
        #     offset = 0 #?
        #     # Print each CSV file
        #     for nscsvfile in csv_files:
        #         print(offset)
        #         df = pd.read_csv(path / nscsvfile)
        #         temp_df = df_MDfeatures.iloc[offset::11]
        #         temp_df.reset_index(drop=True, inplace=True)

        #         df = pd.concat([df, temp_df], axis=1)
        #         df.to_csv(path / nscsvfile, index=False)
        #         print(list(df.columns))
        #         #needs to be dfs and not 1 df
        #         dfs.append(df)
                
        #         offset+=1
        #     reduced_models, df_results = Kfold_Cross_Validation_new(dfs, public_variables.hyperparameter_grid, kfold_=kfold_value, scoring_=scoring_metric)
        #     csv_filename = f'results_K{kandscoring[0]}_{kandscoring[1]}_red_MD.csv'
        #     folder_path = base_path / public_variables.dataframes_folder_red_MD / f'k{kfold_value}_{kandscoring[1]}' / f'{inner_key}'
            
        #     visualize_scores_box_plot(df_results, kfold_value,kandscoring[1], folder_path)
        #     df_results.to_csv(folder_path / csv_filename, index=False)
        #     #dic_models[x].update(reduced_models_dic)
    return



def main():
    base_path = public_variables.base_path_
    
    energyfolder_path = public_variables.energyfolder_path_
    xvgfolder_path = public_variables.xvgfolder_path_
    
    # print('bond')
    # print(df_MDfeatures)
    #get all the models (including the reduced ones)
    folder_for_results_path =  Path(base_path) / public_variables.dfs_reduced_path_ / public_variables.Modelresults_folder_
    #dic_models = randomForest_read_in_models.read_in_model_dictionary(folder_for_results_path, 'original_models_dic.pkl')

    #implement MD features into the right ones. how? #NOTE: via combining the csv files!
    MD_features_implementation(base_path)

    #print(dic_models)
    
    #get_molecules_lists_temp(base_path)

    return

if __name__ == "__main__":
    main()
