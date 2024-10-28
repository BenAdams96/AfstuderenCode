from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import randomForest_read_in_models
import csv_to_dataframes
import randomForest_Class
from randomForest_Class import RandomForestModel
import public_variables

from sklearn.preprocessing import StandardScaler
from csv_to_dataframes import csvfiles_to_dfs
import csv_to_dictionary

import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
from typing import Dict
import numpy as np
from pathlib import Path

import pandas as pd
import math
import re
import os

def Kfold_Cross_Validation_incl_grouped(dfs_in_dic, hyperparameter_grid, kfold_, scoring_):
    print("kfold new + grouped")
    
    columns_ = ["mol_id", "mean_test_score", "std_test_score", "params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_)
    models = {}

    for name, df in dfs_in_dic.items():
        print(name)

        targets = df['PKI']
        unique_mol_ids = df['mol_id'].unique()
        
        kf = KFold(n_splits=kfold_, shuffle=True, random_state=1)
        splits = list(kf.split(unique_mol_ids))
        
        custom_splits = []
        for train_mol_indices, test_mol_indices in kf.split(unique_mol_ids):
            # Get actual molecule IDs for train and test sets
            train_mols = unique_mol_ids[train_mol_indices]
            test_mols = unique_mol_ids[test_mol_indices]
            
            # Map back to the full dataset indices (6000 rows)
            train_indices = df[df['mol_id'].isin(train_mols)].index
            test_indices = df[df['mol_id'].isin(test_mols)].index
            
            # Append as a tuple of arrays
            custom_splits.append((train_indices, test_indices))
        # print('custom splits')
        # print(custom_splits[1][0][0:40])
        # print(custom_splits[1][1][0:40])
        # print(targets[0:40])
        # Initial model and grid search outside the loop
        rf_model = RandomForestModel(n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt')
        rf_model.model.random_state = 42
        df = df.drop(columns=['mol_id','PKI','conformations (ns)'], axis=1, errors='ignore')
        
        #NOTE: careful: conformations (ns) is still in it
        grid_search = rf_model.hyperparameter_tuning(df,targets,hyperparameter_grid,cv=custom_splits,scoring_=scoring_)
        
        df_results = pd.DataFrame(grid_search.cv_results_) #what results? of cv. so the 10 results and 8 rows (each row was a set of hyperparameters)
        
        #grid_search.best_index_ = the index that is best of the x amount of hyperparameter combinations
        result = df_results.loc[grid_search.best_index_, columns_[1:]] #1: because we dont want to include 'time'
        #df_results = dataframe with 'hyperparametercombinations' amount of rows, and each row has mean_test_score and split_test_scores etc
        #result = of the best combination of hyperparameters. mean_test_score, std, params, split 0 to 4
        #make it so that the scores are always positive
        result["mean_test_score"] = abs(result['mean_test_score']) 
        for i in range(kfold_):
            result[f'split{i}_test_score'] = abs(result[f'split{i}_test_score'])

        result_df_row = result.to_frame().T
        result_df_row['mol_id'] = name #add a column of the '1ns' etc
        result_df_row = result_df_row[columns_] #change the order to the one specified above

        #make 1 big dataframe for all the '0ns' '1ns' 'rdkit_min' etc
        ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)
        models[name] = rf_model
    return models, ModelResults_

def strip_dataframes(dataframes,feature_index_list_of_lists):
    stripped_dfs = []
    for idx,df in enumerate(dataframes):
        df_stripped = strip_dataframe(df,feature_index_list_of_lists[idx])
        stripped_dfs.append(df_stripped)
    return stripped_dfs

def strip_dataframe(dataframe,feature_index_list):
    df = dataframe.dropna()
    df.reset_index(drop=True, inplace=True)
    sorted_feature_index_list = sorted(feature_index_list, key=int)
    #print(sorted_feature_index_list)
    stripped_df = df[['mol_id', 'PKI'] + sorted_feature_index_list]
    return stripped_df
#correct! not for the parameter grid search!

    # visualize_scores(results_df, kfold_=kfold_value, scoring_=scoring_value[1], save_plot_folder=folder_for_results_path)
    # visualize_correlation_matrix(correlation_matrix,save_plot_folder,kfold_,scoring_):
def visualize_scores_box_plot(results_df, kfold_,scoring_, save_plot_folder):
    plt.figure(figsize=(12, 6))

    split_columns = [f'split{i}_test_score' for i in range(kfold_)]

    selected_values = results_df.loc[:, split_columns]

    # Convert to list of lists
    selected_values_list = selected_values.values.tolist()
    print("selected_values_list")
    print(selected_values_list)
    x_pos = np.arange(len(results_df))
    print(x_pos)
    plt.boxplot(selected_values_list, patch_artist=True, boxprops=dict(facecolor='skyblue', color='black', alpha=0.7), medianprops=dict(color='red'))
    plt.title(f'{public_variables.RDKIT_descriptors_} descriptors, Kfold = {kfold_}, {scoring_})')
    plt.xlabel('Models at different nanoseconds')
    plt.ylabel('Score')
    plt.xticks(ticks=x_pos+1, labels = ['RD kit' if i == 0 else f'{i-1} ns' for i in x_pos], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.savefig( save_plot_folder / f'boxplot_kfold_{kfold_}_scoring_{scoring_}.png')
    return

def visualize_scores_box_plot_reduced(results_df, kfold_,scoring_, save_plot_folder):
    plt.figure(figsize=(12, 6))
    print(results_df)
    x_pos = np.arange(0,len(results_df))
    print(x_pos)
    plt.boxplot(results_df, patch_artist=True, boxprops=dict(facecolor='skyblue', color='black', alpha=0.7), medianprops=dict(color='red'))
    plt.title(f'(Kfold={kfold_}, Scoring={scoring_})')
    plt.xlabel('NS')
    plt.ylabel('Score')
    plt.xticks(ticks=x_pos+1, labels=[f'ns {i}' for i in x_pos], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.savefig( save_plot_folder / f'boxplotreduced_kfold_{kfold_}_scoring_{scoring_}.png')
    return

def save_originalmodels_hdf5(Modelresults_path, dic_models, dataframes):
    print("save original model hdf5")
    print(dataframes)
    print(type(dataframes[0]))
    hdf5_file = Modelresults_path / 'models_data.h5'
    print(hdf5_file)
    # Saving the original models with the shared dataframe
    with pd.HDFStore(hdf5_file, mode='w') as store:
        for config, models_dict in dic_models.items():
            for model_type, models_list in models_dict.items():
                # Sanitize names
                sanitized_config = config.replace(' ', '_').replace('-', '_').replace('²', '2').replace('(', '').replace(')', '')
                sanitized_model_type = model_type.replace(' ', '_').replace('-', '_')
                
                # Save the list of dataframes for each model
                for i, df in enumerate(dataframes):
                    df_key = f'{sanitized_config}/{sanitized_model_type}/dataframe_{i}'
                    store.put(df_key, df)
                
                # Save the list of original models (convert to string if needed)
                model_key = f'{sanitized_config}/{sanitized_model_type}/models'
                store[model_key] = pd.Series([str(model) for model in models_list])
    return

def save_dataframes_to_csv(dic_with_dfs,save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in dic_with_dfs.items():
        print(f"save dataframe: {name}")
        df.to_csv(save_path / f'{name}.csv', index=False)

def save_modelresults(modelresults, save_path):
    return

def main(dfs_path = public_variables.dfs_descriptors_only_path_):  ###set as default, but can always change it to something else.
    descriptors = public_variables.RDKIT_descriptors_            ## why, just did this but cant remember. for name

    #create folder for storing the models and results from them
    Modelresults_path = dfs_path / public_variables.Modelresults_folder_
    Modelresults_path.mkdir(parents=True, exist_ok=True)


    #l = ['0ns', '1ns', '2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '8ns', '9ns', '10ns','conformations_10','conformations_20','conformations_100','conformations_200','conformations_500','conformations_1000']
    l = ['0ns', '1ns', '2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '8ns', '9ns', '10ns','conformations_10','conformations_20','conformations_100','conformations_200','conformations_500','conformations_1000']

    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic(dfs_path, exclude_files=['concat_hor.csv','concat_ver.csv','conformations_1000_molid.csv','conformations_1000.csv','MD_output.csv']) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    print(dfs_in_dic.keys())
    #remove the mol_id and PKI #NOTE: empty rows have already been removed beforehand, but still do it just to be sure!
    columns_to_drop = ['mol_id', 'PKI', "conformations (ns)"]
    
    #order the keys in the dictionary
    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys())) #RDKIT first
    dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic} #order
    print(sorted_keys_list)
    
    parameter_grid = public_variables.parameter_grid_ #kfolds (5, 10) and metrics (rmse, r2)
    hyperparameter_grid = public_variables.hyperparameter_grid_ #n_estimators, max_depth etc.

    param_combinations = list(itertools.product(parameter_grid['kfold_'], parameter_grid['scoring_']))
    print(param_combinations)
    dic_models = {}

    for kfold_value, scoring_value in param_combinations:
        print(f"kfold_: {kfold_value}, scoring_: {scoring_value[0]}")
        models_dic, Modelresults_ = Kfold_Cross_Validation_incl_grouped(dfs_in_dic, hyperparameter_grid, kfold_=kfold_value, scoring_=scoring_value[0])
        print('done with Kfold cross validation')
        csv_filename = f'results_K{kfold_value}_{scoring_value[1]}_{descriptors}.csv'
        Modelresults_.to_csv(Modelresults_path / csv_filename, index=False)
        
        #instantialize the dic of dictionaries to store all models eventually.
        #TODO: make it so that the dataframe is also stored with it. no. later perhaps
        modelnames = f'RF_Allmodels_k{kfold_value}_{scoring_value[1]}'
        dic_models[modelnames] = {}
        dic_models[modelnames]['original models'] = models_dic
        # save_models(original_models,folder_for_results_path,kfold_value,scoring_value[1])
        # visualize_scores(results_df, kfold_=kfold_value, scoring_=scoring_value[1], save_plot_folder=folder_for_results_path)
        visualize_scores_box_plot(Modelresults_, kfold_value,scoring_value[1], Modelresults_path)
        print(models_dic)
        # save_models_reduced(red_models[0],folder_for_results_path,kfold_value,scoring_value[1])
        # visualize_scores_box_plot_reduced(red_models[1], kfold_value,scoring_value[1], folder_for_results_path)
    print(dic_models)
    
    #print(dic_models['RF_Allmodels_k5_RMSE']['original models']['1.5ns']) to acces a model

    #TODO: general script that contains save models instead of in randomforest_read_in_models
    #save_originalmodels_hdf5(Modelresults_path, dic_models, dfs_stripped)
    randomForest_read_in_models.save_model_dictionary(Modelresults_path,'original_models_dic.pkl',dic_models)
    # return

    return

if __name__ == "__main__":
    # bigdf = pd.read_csv(public_variables.dataframes_master_ / 'conformations_1000_molid.csv')
    # dic = {}
    # targets = bigdf['PKI']
    # print(targets)
    # dic['total_df_ordered_by_moldid.csv'] = bigdf
    # print(dic.keys())
    # Kfold_Cross_Validation_incl_grouped(dic,targets, public_variables.hyperparameter_grid_, 5, 'neg_root_mean_squared_error')
    # main()
    # main(public_variables.dfs_descriptors_only_path_)
    # main(public_variables.dfs_reduced_path_)
    main(public_variables.dfs_reduced_and_MD_path_)
    main(public_variables.dfs_MD_only_path_)

