from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import randomForest_read_in_models

from Models.RF.RF_Class import RandomForestModel
from Models.XGB.XGBoost_Class import XGBoostModel
from Models.SVM.SVM_class import SupportVectorMachineRegressor
from global_files import public_variables, csv_to_dictionary

from sklearn.preprocessing import StandardScaler

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

def visualize_splits(custom_split, df, output_dir="splits_visualization"):
    """
    Visualizes the distribution of the PKI column for training and test sets 
    in the custom splits and saves histogram plots for each split.
    
    Parameters:
    - custom_split: list of tuples [(train_indices, test_indices), ...]
    - df: DataFrame containing the target column 'PKI'
    - output_dir: Directory to save the histogram plots (default: "splits_visualization")
    """
    import os
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, (train_indices, test_indices) in enumerate(custom_split):
        # Extract training and test targets
        train_targets = df.loc[train_indices, 'PKI']
        test_targets = df.loc[test_indices, 'PKI']

        # Plot the distributions
        plt.figure(figsize=(10, 6))
        plt.hist(train_targets, bins=20, alpha=0.7, label="Training Set", color="blue")
        plt.hist(test_targets, bins=20, alpha=0.7, label="Test Set", color="orange")
        plt.xlabel("PKI")
        plt.ylabel("Frequency")
        plt.title(f"Split {i+1}: PKI Distribution")
        plt.legend()
        
        # Save the plot
        plt.savefig(f"{output_dir}/split_{i+1}_pki_distribution.png")
        plt.close()

def Kfold_Cross_Validation_incl_grouped_RF(df, hyperparameter_grid, kfold_, scoring_):
    print("kfold new + grouped")
    
    columns_ = ["mol_id", "mean_test_score", "std_test_score", "params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_)
    models = {}

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
    # print(custom_splits[3][0])
    # print(custom_splits[3][1])

    visualize_splits(custom_splits, df, output_dir=public_variables.base_path_)
    # print('custom splits')
    # print(custom_splits[1][0][0:40])
    # print(custom_splits[1][1][0:40])
    # print(targets[0:40])
    # Initial model and grid search outside the loop
    rf_model = RandomForestModel(n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt')
    rf_model.model.random_state = 42
    df = df.drop(columns=['mol_id','PKI','conformations (ns)'], axis=1, errors='ignore')
    print(df)
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
    result_df_row['mol_id'] = '2D' #add a column of the '1ns' etc
    result_df_row = result_df_row[columns_] #change the order to the one specified above

    #make 1 big dataframe for all the '0ns' '1ns' 'rdkit_min' etc
    ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)
    models['2D'] = rf_model
    return models, ModelResults_

def Kfold_Cross_Validation_incl_grouped_XGB(df, hyperparameter_grid, kfold_, scoring_):
    print("kfold new + grouped")
    
    columns_ = ["mol_id", "mean_test_score", "std_test_score", "params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_)
    models = {}

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
    rf_model = XGBoostModel()
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
    result_df_row['mol_id'] = 'XGB' #add a column of the '1ns' etc
    result_df_row = result_df_row[columns_] #change the order to the one specified above

    #make 1 big dataframe for all the '0ns' '1ns' 'rdkit_min' etc
    ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)
    models['XGB'] = rf_model
    return models, ModelResults_

def Kfold_Cross_Validation_incl_grouped_SVM(df, hyperparameter_grid, kfold_, scoring_):
    print("kfold new + grouped")
    
    columns_ = ["mol_id", "mean_test_score", "std_test_score", "params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_)
    models = {}


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
    # Initialize SVM model
    print(custom_splits)
    svm_model = SupportVectorMachineRegressor(C=1.0, kernel='rbf')  # Default values, customize as needed

    # Drop columns not used in features
    df_features = df.drop(columns=['mol_id', 'PKI', 'conformations (ns)'], axis=1, errors='ignore')

    # Perform grid search with the SVM model
    grid_search = svm_model.hyperparameter_tuning(df_features, targets, hyperparameter_grid, cv=custom_splits, scoring_=scoring_)

    # Collect results
    df_results = pd.DataFrame(grid_search.cv_results_)
    result = df_results.loc[grid_search.best_index_, columns_[1:]]
    
    # Make scores positive
    result["mean_test_score"] = abs(result['mean_test_score'])
    for i in range(kfold_):
        result[f'split{i}_test_score'] = abs(result[f'split{i}_test_score'])

    result_df_row = result.to_frame().T
    result_df_row['mol_id'] = 'SVM'
    result_df_row = result_df_row[columns_]

    # Concatenate results for all mol_id categories
    ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)
    models['SVM'] = svm_model
    
    return models, ModelResults_

def main(dataframes_master):  ###set as default, but can always change it to something else.
    csv_file_path =  dataframes_master_ / '2D_ECFP.csv'

    #create folder for storing the models and results from them
    Modelresults_path_RF = dataframes_master / f'ModelResults_RF'
    Modelresults_path_RF.mkdir(parents=True, exist_ok=True)
    Modelresults_path_XGB = dataframes_master / f'ModelResults_XGB'
    Modelresults_path_XGB.mkdir(parents=True, exist_ok=True)
    Modelresults_path_SVM = dataframes_master / f'ModelResults_SVM'
    Modelresults_path_SVM.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file_path)

    #remove the mol_id and PKI #NOTE: empty rows have already been removed beforehand, but still do it just to be sure!
    columns_to_drop = ['mol_id', 'PKI', "conformations (ns)"]

    # parameter_grid = public_variables.parameter_grid_
    parameter_grid = {
            'kfold_': [10],
            'scoring_': [('neg_root_mean_squared_error','RMSE')],
            } #kfolds (5, 10) and metrics (rmse, r2)
    hyperparameter_grid_RF = {
            'n_estimators': [50,100,150],
            'max_depth': [8,15],
            'min_samples_split': [8,15],
            'min_samples_leaf': [8,15],
            'max_features': ['sqrt'] #'None' can lead to overfitting.
        }
    
    # # hyperparameter_grid_XGboost = {
    #     'n_estimators': [50, 100, 150],          # Number of trees (lower values for quicker training)
    #     'max_depth': [3, 5, 7],             # Maximum depth of each tree (shallower trees to avoid overfitting)
    #     'learning_rate': [0.01, 0.1],       # Learning rate (smaller values for more gradual training)
    #     'subsample': [0.6, 0.8],            # Subsample ratio of the training instance (to prevent overfitting)
    #     'colsample_bytree': [0.6, 0.8],     # Subsample ratio of columns when constructing each tree
    #     'gamma': [0, 0.1],                  # Minimum loss reduction required to make a further partition on a leaf node
    # }
    hyperparameter_grid_XGB = {
    'n_estimators': [50, 100],                # Limit the number of boosting rounds
    'max_depth': [3, 5],                       # Shallower trees to prevent overfitting
    'learning_rate': [0.01, 0.05],             # Lower learning rate to control overfitting
    'subsample': [0.6, 0.8],                   # Limit subsampling to reduce overfitting risk
    'colsample_bytree': [0.6, 0.8],            # Limit feature subsampling
    'gamma': [0, 0.1],                         # Keep gamma small to avoid excessive pruning
    }

    hyperparameter_grid_SVM = {
    'kernel': ['rbf','linear'],        # Most commonly effective kernels
    'C': [0.1, 1,10],                      # Reasonable regularization strengths for regression
    'epsilon': [0.1],             # Standard values for error tolerance
    'gamma': ['scale', 0.1]             # Default and a specific small value for tuning influence
    }
    
    param_combinations = list(itertools.product(parameter_grid['kfold_'], parameter_grid['scoring_']))
    print(param_combinations)
    dic_models = {}

    for kfold_value, scoring_value in param_combinations:
        print(f"kfold_: {kfold_value}, scoring_: {scoring_value[0]}")
        models_dic, Modelresults_ = Kfold_Cross_Validation_incl_grouped_RF(df, hyperparameter_grid_RF, kfold_=kfold_value, scoring_=scoring_value[0])
        print('done with Kfold cross validation')
        csv_filename = f'results_K{kfold_value}_{scoring_value[1]}_2D.csv'
        Modelresults_.to_csv(Modelresults_path_RF / csv_filename, index=False)


        models_dic, Modelresults_ = Kfold_Cross_Validation_incl_grouped_XGB(df, hyperparameter_grid_XGB, kfold_=kfold_value, scoring_=scoring_value[0])
        print('done with Kfold cross validation')
        csv_filename = f'results_K{kfold_value}_{scoring_value[1]}_2D.csv'
        Modelresults_.to_csv(Modelresults_path_XGB / csv_filename, index=False)


        models_dic, Modelresults_ = Kfold_Cross_Validation_incl_grouped_SVM(df, hyperparameter_grid_SVM, kfold_=kfold_value, scoring_=scoring_value[0])
        print('done with Kfold cross validation')
        csv_filename = f'results_K{kfold_value}_{scoring_value[1]}_2D.csv'
        Modelresults_.to_csv(Modelresults_path_SVM / csv_filename, index=False)
        
        #instantialize the dic of dictionaries to store all models eventually.
        #TODO: make it so that the dataframe is also stored with it. no. later perhaps
        
        # save_models(original_models,folder_for_results_path,kfold_value,scoring_value[1])
        # visualize_scores(results_df, kfold_=kfold_value, scoring_=scoring_value[1], save_plot_folder=folder_for_results_path)

        # modelnames = f'RF_Allmodels_k{kfold_value}_{scoring_value[1]}'
        # dic_models[modelnames] = {}
        # dic_models[modelnames]['original models'] = models_dic
        # visualize_scores_box_plot(Modelresults_, kfold_value,scoring_value[1], Modelresults_path)
        # print(models_dic)

        # save_models_reduced(red_models[0],folder_for_results_path,kfold_value,scoring_value[1])
        # visualize_scores_box_plot_reduced(red_models[1], kfold_value,scoring_value[1], folder_for_results_path)
    # print(dic_models)

    #print(dic_models['RF_Allmodels_k5_RMSE']['original models']['1.5ns']) to acces a model

    #TODO: general script that contains save models instead of in randomforest_read_in_models
    #save_originalmodels_hdf5(Modelresults_path, dic_models, dfs_stripped)
    # randomForest_read_in_models.save_model_dictionary(Modelresults_path,'original_models_dic.pkl',dic_models)
    # return

    return

if __name__ == "__main__":
    dataframes_master_ = public_variables.base_path_ / Path(f'2D_models')

    main(dataframes_master_)

