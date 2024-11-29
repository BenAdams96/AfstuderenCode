from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import randomForest_read_in_models
import csv_to_dataframes

from A_randomForest_Class import RandomForestModel
import public_variables

from sklearn.preprocessing import StandardScaler
from csv_to_dataframes import csvfiles_to_dfs
import csv_to_dictionary
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
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

def save_dataframes_to_csv(dic_with_dfs,save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in dic_with_dfs.items():
        print(f"save dataframe: {name}")
        df.to_csv(save_path / f'{name}.csv', index=False)








def bin_pki_values(y, num_bins=5):
    """
    Bin the pKi values into discrete bins for stratified splitting.
    Parameters:
        y (array-like): Target variable (pKi values).
        num_bins (int): Number of bins to stratify.
    Returns:
        binned_y: Binned version of y.
    """
    bins = np.linspace(y.min(), y.max(), num_bins + 1)  # Define bin edges
    print(bins)
    binned_y = pd.cut(y, bins, right=True, include_lowest=True, labels=False)  # Include both sides
    return binned_y

def visualize_folds_distribution(y, fold_indices, title="pKi Distribution Across Folds"):
    """
    Visualize the distribution of pKi values across folds.
    Parameters:
        y (array-like): Target variable (pKi values).
        fold_indices (list of tuples): List of (train_idx, test_idx) for each fold.
    """
    plt.figure(figsize=(10, 6))
    for fold_num, (train_idx, test_idx) in enumerate(fold_indices, 1):
        train_values = y[train_idx]
        test_values = y[test_idx]
        plt.hist(train_values, bins=10, alpha=0.5, label=f"Fold {fold_num} Train", histtype='step')
        plt.hist(test_values, bins=10, alpha=0.5, label=f"Fold {fold_num} Test", histtype='stepfilled')
    
    plt.xlabel("pKi Values")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.savefig('save.png')

def nested_cross_validation(df, target_column, outer_folds=5, inner_folds=5):
    """
    Perform nested cross-validation with stratified outer folds.
    Parameters:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        outer_folds (int): Number of outer folds.
        inner_folds (int): Number of inner folds.
    """
    print("kfold new + grouped")
    # df = df.set_index('mol_id')
    # df = df.sort_values(by='mol_id')
    print(df)
    X = df.drop(columns=[target_column, 'mol_id', 'conformations (ns)'], axis=1, errors='ignore')
    
    y = df[target_column]  # Target (pKi values)

    # Stratified outer loop
    binned_y = bin_pki_values(y, num_bins=5)  # Bin pKi values for stratification, still a dataframe: all molecules with an index of which bin they are in
    unique, counts = np.unique(binned_y, return_counts=True)
    print(counts)

    # groups = df.index
    groups = df['mol_id']
    print('groups')
    print(groups)
    outer_cv = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    fold_results = []
    fold_indices = []  # To store fold indices for visualization
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, binned_y, groups=groups)):
        print(f'outer fold: {outer_fold}')
        # print(df.iloc[train_idx])
        # print(df.iloc[test_idx])
        fold_indices.append((train_idx, test_idx))  # Store for visualization

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] #all still a dataframe
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = X_train.index #Index([1,3,4,...]) these are the idx but in mol_id
        groups_train_y = y_train.index
        binned_y_train = binned_y.iloc[train_idx]
        # print(groups_train.tolist())
        print('test')
        print(sorted(df.iloc[groups_train.tolist()]['mol_id'].tolist()))
        print(sorted(df.iloc[groups_train_y.tolist()]['mol_id'].tolist()))
        # Inner loop CV for hyperparameter tuning
        best_model = None
        best_score = -np.inf

        inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=42)
        custom_splits_for_inner_validation = []

        for inner_fold,(inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train, binned_y_train, groups=groups_train)):
            print(f'inner fold: {inner_fold}')
            X_train_inner, X_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
            y_train_inner, y_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]
            print(X_val)
            # Keep track of the original indices in the inner loop
            original_train_inner_indices = X_train_inner.index
            original_val_indices = X_val.index
            print(original_val_indices)
            print(original_train_inner_indices.tolist())
            all_train_indices = df[df.index.isin(original_train_inner_indices)].index
            all_val_indices = df[df.index.isin(original_val_indices)].index #with dataframe where mol_id happens multiple times, the dataframe grows, but its not actually idx, its mol id 1 1 1 1 1 for example

            custom_splits_for_inner_validation.append((original_train_inner_indices, original_val_indices))

            print(all_val_indices.tolist())
            print(inner_train_idx)

#         
#         
#             # Get actual molecule IDs for train and test sets
#             train_mols = unique_mol_ids[train_mol_indices]
#             test_mols = unique_mol_ids[test_mol_indices]
            
#             # Map back to the full dataset indices (6000 rows)
#             train_indices = df[df['mol_id'].isin(train_mols)].index
#             test_indices = df[df['mol_id'].isin(test_mols)].index
            
#             # Append as a tuple of arrays
#             custom_splits.append((train_indices, test_indices))
#         # print('custom splits')
#         # print(custom_splits[1][0][0:40])
#         # print(custom_splits[1][1][0:40])
#         # print(targets[0:40])
#         # Initial model and grid search outside the loop
#         rf_model = RandomForestModel(n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt')
#         rf_model.model.random_state = 42
#         df = df.drop(columns=['mol_id','PKI','conformations (ns)'], axis=1, errors='ignore')
        
#         grid_search = rf_model.hyperparameter_tuning(df,targets,hyperparameter_grid,cv=custom_splits,scoring_=scoring_)





        # inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=42)
        # for params in [
        #     dict(zip(public_variables.param_grid.keys(), values))
        #     for values in itertools.product(*param_grid.values())
        # ]:
        #     inner_scores = []
            
        #     for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
        #         X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
        #         y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]
                
        #         # Train the model
        #         model = RandomForestModel(**params)
        #         model.fit(X_inner_train, y_inner_train)
                
        #         # Validate
        #         y_val_pred = model.predict(X_val)
        #         score = r2_score(y_val, y_val_pred)  # Use R² for evaluation
        #         inner_scores.append(score)
            
        #     mean_inner_score = np.mean(inner_scores)
        #     if mean_inner_score > best_score:
        #         best_score = mean_inner_score
        #         best_model = RandomForestModel(**params)
        
        # # Train best model on the full outer training set
        # best_model.fit(X_train, y_train)
        
        # # Test on the outer fold
        # y_test_pred = best_model.predict(X_test)
        # fold_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        # fold_r2 = r2_score(y_test, y_test_pred)
        
        # fold_results.append({
        #     "Outer Fold": outer_fold,
        #     "Best Params": best_model.__repr__(),
        #     "RMSE": fold_rmse,
        #     "R²": fold_r2
        # })
    



    # Visualize pKi distribution in folds
    # visualize_folds_distribution(y, fold_indices)
    
    return pd.DataFrame(fold_results)


def main(dfs_path = public_variables.dfs_descriptors_only_path_):  ###set as default, but can always change it to something else.
    descriptors = public_variables.RDKIT_descriptors_            ## why, just did this but cant remember. for name

    #create folder for storing the models and results from them
    Modelresults_path = dfs_path / public_variables.Modelresults_folder_
    Modelresults_path.mkdir(parents=True, exist_ok=True)

    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic(dfs_path, exclude_files=['0ns.csv', '2ns.csv', '3ns.csv', '4ns.csv', '5ns.csv', '6ns.csv', '7ns.csv', '8ns.csv', '9ns.csv', '10ns.csv','conformations_1000_molid.csv','MD_output.csv','conformations_200.csv','conformations_50.csv']) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    print(dfs_in_dic.keys())
    #remove the mol_id and PKI #NOTE: empty rows have already been removed beforehand, but still do it just to be sure!
    columns_to_drop = ['mol_id', 'PKI', "conformations (ns)"]
    
    # #order the keys in the dictionary
    # sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys())) #RDKIT first
    # dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic} #order
    # # print(sorted_keys_list)
    
    parameter_grid = public_variables.parameter_grid_ #kfolds (5, 10) and metrics (rmse, r2)
    hyperparameter_grid = public_variables.hyperparameter_grid_ #n_estimators, max_depth etc.

    # param_combinations = list(itertools.product(parameter_grid['kfold_'], parameter_grid['scoring_']))
    # print(dfs_in_dic.keys())
    # print(param_combinations)
    dic_models = {}

    df = dfs_in_dic['1ns']
    df = dfs_in_dic['conformations_10']
    df = dfs_in_dic['conformations_20']
    df = dfs_in_dic['conformations_100']
    # df = dfs_in_dic['conformations_500']
    # df = dfs_in_dic['conformations_11']

    mol_id_counts = df['mol_id'].value_counts()
    mol_ids_not_20 = mol_id_counts[mol_id_counts != 20]
    print(mol_ids_not_20)



    # mol_615_rows = df[df['mol_id'] == 615]

    # # Extract the corresponding 'conformations (ns)' values
    # conformations_ns_values = mol_615_rows['conformations (ns)']

    # # Display the result
    # print(f"Conformations (ns) values for mol_id 615:")
    # print(conformations_ns_values.tolist())
    fold_results = nested_cross_validation(df,'PKI')

    # for kfold_value, scoring_value in param_combinations:
    #     print(f"kfold_: {kfold_value}, scoring_: {scoring_value[0]}")
    #     models_dic, Modelresults_ = Kfold_Cross_Validation_incl_grouped(dfs_in_dic, hyperparameter_grid, kfold_=kfold_value, scoring_=scoring_value[0])
    #     print('done with Kfold cross validation')
    #     csv_filename = f'results_K{kfold_value}_{scoring_value[1]}_{descriptors}.csv'
    #     Modelresults_.to_csv(Modelresults_path / csv_filename, index=False)
        
    #     #instantialize the dic of dictionaries to store all models eventually.
    #     #TODO: make it so that the dataframe is also stored with it. no. later perhaps
    #     modelnames = f'RF_Allmodels_k{kfold_value}_{scoring_value[1]}'
    #     dic_models[modelnames] = {}
    #     dic_models[modelnames]['original models'] = models_dic
    #     # save_models(original_models,folder_for_results_path,kfold_value,scoring_value[1])
    #     # visualize_scores(results_df, kfold_=kfold_value, scoring_=scoring_value[1], save_plot_folder=folder_for_results_path)
    #     visualize_scores_box_plot(Modelresults_, kfold_value,scoring_value[1], Modelresults_path)
    #     print(models_dic)
    #     # save_models_reduced(red_models[0],folder_for_results_path,kfold_value,scoring_value[1])
    #     # visualize_scores_box_plot_reduced(red_models[1], kfold_value,scoring_value[1], folder_for_results_path)
    # print(dic_models)
    
    # #print(dic_models['RF_Allmodels_k5_RMSE']['original models']['1.5ns']) to acces a model

    # #TODO: general script that contains save models instead of in randomforest_read_in_models
    # #save_originalmodels_hdf5(Modelresults_path, dic_models, dfs_stripped)
    # randomForest_read_in_models.save_model_dictionary(Modelresults_path,'original_models_dic.pkl',dic_models)
    # # return

    return

if __name__ == "__main__":




    main(public_variables.dfs_descriptors_only_path_)
    # main(public_variables.dfs_PCA_path)


    # main(public_variables.dfs_reduced_and_MD_path_)
    # main(public_variables.dfs_MD_only_path_)

    # main(public_variables.dataframes_master_/'reduced_t0.85')
    # main(public_variables.dataframes_master_/'descriptors only scaled mw')

