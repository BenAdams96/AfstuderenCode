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

def standardize_dataframe(df):
    """Standardize the dataframe features."""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df)
    standardized_df = pd.DataFrame(standardized_data, columns=df.columns)
    return standardized_df

def Kfold_Cross_Validation(dfs_in_dic,targets, hyperparameter_grid, kfold_, scoring_): #add targets
    print("kfold new")
    
    columns_ = ["id","mean_test_score","std_test_score","params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_) #NOTE needs to be a pd dataframe
    models: Dict[str, RandomForestModel]= {} #[str, RandomForestModel] contains the type of keys and values
    for name, df in dfs_in_dic.items():
        
        X_train = df
        
        kf = KFold(n_splits=kfold_, shuffle=True, random_state=1)
        
        rf_model = RandomForestModel(n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt')
        rf_model.model.random_state = 42
        
        grid_search = rf_model.hyperparameter_tuning(X_train,targets,hyperparameter_grid,cv=kf,scoring_=scoring_)
        
        df_results = pd.DataFrame(grid_search.cv_results_) #what results? of cv. so the 10 results
        #grid_search.best_index_ = the index that is best of the x amount of hyperparameter combinations
        result = df_results.loc[grid_search.best_index_, columns_[1:]] #1: because we dont want to include 'time'
        #df_results = dataframe with 'hyperparametercombinations' amount of rows, and each row has mean_test_score and split_test_scores etc
        #result = of the best combination of hyperparameters. mean_test_score, std, params, split 0 to 4
        #make it so that the scores are always positive
        result["mean_test_score"] = abs(result['mean_test_score']) 
        for i in range(kfold_):
            result[f'split{i}_test_score'] = abs(result[f'split{i}_test_score'])

        result_df_row = result.to_frame().T
        result_df_row['id'] = name #add a column of the '1ns' etc
        result_df_row = result_df_row[columns_] #change the order to the one specified above

        #make 1 big dataframe for all the '0ns' '1ns' 'rdkit_min' etc
        ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)
        models[name] = rf_model
    return models, ModelResults_


def reduced_model(rf_model,X_train,y_train,kf,scoring):
    importances = rf_model.top_features #still in order of feature 1 2 3 4 ...
            
    sorted_indices = importances.index.tolist() #list of indices of most important features first [28, 71, 16 ... ] corresponding to indices in 'importances' with highest features
    cumulative_importance = np.cumsum(importances[sorted_indices]) * 100 #list of the sum of the importances.
    
    threshold = 75
    top_indices = cumulative_importance[cumulative_importance <= threshold].index.tolist()
    cv_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Extract top 25 features for current fold
        X_train_fold_top25 = X_train_fold[top_indices]
        X_val_fold_top25 = X_val_fold[top_indices]
        
        # Initialize and fit RandomForestRegressor
        rf_model_reduced = RandomForestModel(n_trees=rf_model.n_trees,
                                   max_depth=rf_model.max_depth,
                                   min_samples_split=rf_model.min_size,
                                   max_features=rf_model.max_features)
        rf_model_reduced.fit(X_train_fold_top25, y_train_fold)
        
        # Evaluate on validation set
        rmse, mse, r2 = rf_model_reduced.evaluate(X_val_fold_top25, y_val_fold)
        
        # Store or print the RMSE for current fold
        if scoring.lower().strip() == 'neg_root_mean_squared_error':
            cv_scores.append(rmse)
        elif scoring.lower().strip() == 'r2':
            cv_scores.append(r2)
        else:
            print("NO SCORING: ERROR")
    return rf_model_reduced, cv_scores


def save_models_reduced(modellist,save_path,k,scoring):
    # Full path to the file using pathlib.Path
    all_models_path = save_path / f'RF_reduced_Allmodels_k{k}_{scoring}.pkl'

    with open(all_models_path, 'wb') as file:
        pickle.dump(modellist, file)
    return


def visualize_correlation_matrix(correlation_matrix,save_plot_folder,kfold_,scoring_):
    """Visualize the correlation matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.tight_layout()
    plt.savefig( save_plot_folder / f'plot_kfold_{kfold_}_scoring_{scoring_}.png')
    plt.show()

def visualize_scores(results_df, kfold_,scoring_, save_plot_folder):
    plt.figure(figsize=(12, 6))

    x_pos = np.arange(len(results_df))
    bars = plt.bar(x_pos, results_df['mean_test_score'], yerr=results_df['std_test_score'], capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Mean Test Scores with Standard Deviations (Kfold={kfold_}, Scoring={scoring_})')
    plt.xlabel('Index')
    plt.ylabel('Mean Test Score')
    plt.xticks(ticks=x_pos, labels=[f'Model {i}' for i in x_pos], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.savefig( save_plot_folder / f'barplot_kfold_{kfold_}_scoring_{scoring_}.png')
    return
#read in dataframe

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

def strip_dataframes(df):
    
    return df

def main(dfs_path = public_variables.dfs_descriptors_only_path_):  ###set as default, but can always change it to something else.
    descriptors = public_variables.RDKIT_descriptors_            ## why, just did this but cant remember. for name

    #create folder for storing the models and results from them
    Modelresults_path = dfs_path / public_variables.Modelresults_folder_
    Modelresults_path.mkdir(parents=True, exist_ok=True)


    dfs_descriptors_only_path = public_variables.dfs_descriptors_only_path_ # e.g., 'dataframes_JAK1_WHIM_i1'
    dfs_reduced_path = public_variables.dfs_reduced_path_  # e.g., 'dataframes_JAK1_WHIM_i1_t0.85'

    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic(dfs_path, exclude_files=['concat_hor.csv','concat_ver.csv']) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    
    #remove the mol_id and PKI #NOTE: empty rows have already been removed beforehand, but still do it just to be sure!
    columns_to_drop = ['mol_id', 'PKI']
    
    #order the keys in the dictionary
    sorted_keys_list = csv_to_dictionary.get_sorted_folders_namelist(list(dfs_in_dic.keys())) #RDKIT first
    dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic} #order
    print(sorted_keys_list)

    #clean dataframes just to be sure, also remove csv files if they dont have 'PKI' column + change the order correctly
    for name, df in list(dfs_in_dic.items()):  # Use list() to avoid runtime dictionary size change
        df = df.dropna()
        if 'PKI' in df.columns:  # Check if 'PKI' column exists
            targets = df['PKI']
        else:
            print('else')
            del dfs_in_dic[name]  # Remove DataFrame if 'PKI' column is not found
            continue
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])  # Avoid inplace=True
        df = df.reset_index(drop=True)  # Avoid inplace=True
        dfs_in_dic[name] = df
    
    parameter_grid = public_variables.parameter_grid_ #kfolds (5, 10) and metrics (rmse, r2)
    hyperparameter_grid = public_variables.hyperparameter_grid_ #n_estimators, max_depth etc.

    param_combinations = list(itertools.product(parameter_grid['kfold_'], parameter_grid['scoring_']))

    dic_models = {}

    for kfold_value, scoring_value in param_combinations:
        print(f"kfold_: {kfold_value}, scoring_: {scoring_value[0]}")
        models_dic, Modelresults_ = Kfold_Cross_Validation(dfs_in_dic, targets, hyperparameter_grid, kfold_=kfold_value, scoring_=scoring_value[0])
        print(models_dic)
        #TODO: make it so that it does 5 times the kfold so we have plenty of different models to get a good average
        csv_filename = f'results_K{kfold_value}_{scoring_value[1]}_{descriptors}.csv'
        Modelresults_.to_csv(Modelresults_path / csv_filename, index=False)
        
        #instantialise the dic of dictionaries to store all models eventually.
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
    
    main()
