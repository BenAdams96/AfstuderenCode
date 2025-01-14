from pathlib import Path
#basic paths
base_path_ = Path(__file__).resolve().parent.parent.parent

print(base_path_)

#dataset related variable names
MLmodel_ = 'RF'
Descriptor_ = 'WHIM' #choose between 'WHIM' and 'GETAWAY' #VARIABLE
dataset_protein_ = 'JAK1'   #VARIABLE 'JAK1' or 'GSK3'
dataset_filename_ = dataset_protein_ + '_dataset.csv'
dataset_path_ = base_path_ / dataset_filename_

timeinterval_snapshots = 1 #in ns #VARIABLE (smallest value possible =  0.01 ns which is equal to 10ps)
correlation_threshold_ = 0.85

#MD simulations folder file
MDsimulations_folder_ = f'MDsimulations_{dataset_protein_}'
MDsimulations_path_ = base_path_.parents[0] / MDsimulations_folder_

#prepare_energy_files
energyfolder_path_ = base_path_ / f'energyfolder_files_{dataset_protein_}'
edrfolder_path_ = energyfolder_path_ / f'edr_files_{dataset_protein_}'
xvgfolder_path_ = energyfolder_path_ / f'xvg_files_MD_features_{dataset_protein_}'
# MDfeatures_allmol_csvfile = f'MDfeatures_for_all_molecules_{dataset_protein_}_{timeinterval_snapshots}ns.csv'
MDfeatures_allmol_csvfile = f'MD_features_{dataset_protein_}.csv'

ligand_conformations_folder_ = f'ligand_conformations_{dataset_protein_}'
ligand_conformations_path_ = base_path_ / ligand_conformations_folder_

dataframes_master_ = base_path_ / Path(f'dataframes_{dataset_protein_}_{Descriptor_}')
initial_dataframe = dataframes_master_ / Path('initial_dataframe.csv')
dfs_descriptors_only_path_ =  dataframes_master_ / 'descriptors only'
dfs_reduced_path_ = dataframes_master_ / f'reduced_t{correlation_threshold_}'
dfs_PCA_path =  dataframes_master_ / 'PCA'
dfs_reduced_and_MD_path_ = dataframes_master_ / f'reduced_t{correlation_threshold_}_MD'
dfs_MD_only_path_ = dataframes_master_ / 'MD only'
Modelresults_folder_ = f'ModelResults_{MLmodel_}' #not a path because can be in different paths
Modelresults_combined_folder_ = f'ModelResults_combined_{MLmodel_}'

# dataframes_folder_red_ = f'dataframes_{dataset_protein_}_{descriptors_}_i{timeinterval_snapshots}_t{correlation_threshold_}'
# dataframes_folder_red_MD = f'dataframes_{dataset_protein_}_{descriptors_}_i{timeinterval_snapshots}_t{correlation_threshold_}_MD'
# Modelresults_ = f'ModelResults_{model_}_{dataset_protein_}_{descriptors_}_i{timeinterval_snapshots}'

# dataframes_folder = 'dataframesWHIMJAK1_with0.95'
# dataframes_folder_red = 'dataframesWHIMJAK1_reduced_with0.95'
# dataframes_folder_red_MD = 'dataframesWHIMJAK1_red_MD_with0.95_good'
# Modelresults_RF = 'ModelResults_RF_095'

#NOTE: features is the one that is used. for prepare_energy_files.py
featuresold = ["Bond","U-B","Proper-Dih.","Improper-Dih.","LJ-(SR)","Coulomb-(SR)","Potential","Total-Energy", "Enthalpy"]
features = ["Bond","U-B","Proper-Dih."]
#NOTE: options:
    # 1  Bond             2  U-B              3  Proper-Dih.      4  Improper-Dih. 
    # 5  LJ-14            6  Coulomb-14       7  LJ-(SR)          8  Disper.-corr. 
    # 9  Coulomb-(SR)    10  Coul.-recip.    11  Potential       12  Kinetic-En.   
    # 13  Total-Energy    14  Conserved-En.   15  Temperature     16  Pres.-DC      
    # 17  Pressure        18  Constr.-rmsd    19  Box-X           20  Box-Y         
    # 21  Box-Z           22  Volume          23  Density         24  pV            
    # 25  Enthalpy        26  Vir-XX          27  Vir-XY          28  Vir-XZ        
    # 29  Vir-YX          30  Vir-YY          31  Vir-YZ          32  Vir-ZX        
    # 33  Vir-ZY          34  Vir-ZZ          35  Pres-XX         36  Pres-XY       
    # 37  Pres-XZ         38  Pres-YX         39  Pres-YY         40  Pres-YZ       
    # 41  Pres-ZX         42  Pres-ZY         43  Pres-ZZ         44  #Surf*SurfTen 
    # 45  T-Other         46  T-SOL
    #NOTE: not every molecule contains IMPROPER-DIH!
# features_all = features = [
#     'Bond', 'U-B', 'Proper-Dih.', 
#     'LJ-14', 'Coulomb-14', 'LJ-(SR)', 'Disper.-corr.', 
#     'Coulomb-(SR)', 'Coul.-recip.', 'Potential', 'Kinetic-En.', 
#     'Total-Energy', 'Conserved-En.', 'Temperature', 'Pres.-DC', 
#     'Pressure', 'Constr.-rmsd', 'Box-X', 'Box-Y', 
#     'Box-Z', 'Volume', 'Density', 'pV', 
#     'Enthalpy', 'Vir-XX', 'Vir-XY', 'Vir-XZ', 
#     'Vir-YX', 'Vir-YY', 'Vir-YZ', 'Vir-ZX', 
#     'Vir-ZY', 'Vir-ZZ', 'Pres-XX', 'Pres-XY', 
#     'Pres-XZ', 'Pres-YX', 'Pres-YY', 'Pres-YZ', 
#     'Pres-ZX', 'Pres-ZY', 'Pres-ZZ', '#Surf*SurfTen', 
#     'T-Other', 'T-SOL']


# dataframes_folder = 'dataframesWHIMJAK1'
# dataframes_folder_red = 'dataframesWHIMJAK1_reduced'
# dataframes_folder_red_MD = 'dataframesWHIMJAK1_red_MD'
# Modelresults_RF = 'ModelResults_RF'

reduced_models_to_create_ = {
        'feature_importance': [0.03],
        'x_features': [15,20,25]
        }

parameter_grid_ = {
        'kfold_': [10],
        'scoring_': [('r2','R-squared (R²)')] # ('neg_root_mean_squared_error','RMSE'),
        }

# parameter_grid = {
    #         'kfold_': [5,10],
    #         'scoring_': [('neg_root_mean_squared_error','RMSE'),('r2','R-squared (R²)')],
    #         }

hyperparameter_grid_ = {
            'n_estimators': [150],
            'max_depth': [8,15],
            'min_samples_split': [10,20],
            'min_samples_leaf': [5],
            'max_features': ['sqrt'] #'None' can lead to overfitting.
        }
# hyperparameter_grid_RF = {
#     'n_estimators': [100, 200],
#     'max_depth': [8, 15],
#     'min_samples_split': [2, 10],
#     'min_samples_leaf': [2, 5],
#     'max_features': ['sqrt']
# }
hyperparameter_grid_RF = {
    'n_estimators': [100],
    'max_depth': [5,8],
    'min_samples_split': [2,5],
    'min_samples_leaf': [2,5],
    'max_features': ['sqrt']
}
# hyperparameter_grid_ = {
#             'n_estimators': [100],
#             'max_depth': [10],
#             'min_samples_split': [10],
#             'min_samples_leaf': [5],
#             'max_features': ['sqrt'] #'None' can lead to overfitting.
#         }

# hyperparameter_grid_ = {
#             'n_estimators': [100,150,300],
#             'max_depth': [10, 30, 50],
#             'min_samples_split': [10, 25, 4],
#             'min_samples_leaf': [5, 10, 15],
#             'max_features': ['sqrt'] #'None' can lead to overfitting.
#         }

hyperparameter_grid_XGB = {
    'n_estimators': [100,150],          # Number of trees (lower values for quicker training)
    'max_depth': [2,5],            # Maximum depth of each tree (shallower trees to avoid overfitting)
    'learning_rate': [0.2],       # Learning rate (smaller values for more gradual training)
    'subsample': [0.7],            # Subsample ratio of the training instance (to prevent overfitting)
    'colsample_bytree': [0.7],     # Subsample ratio of columns when constructing each tree
    'gamma': [0.1],                  # Minimum loss reduction required to make a further partition on a leaf node
}

# hyperparameter_grid_SVM = {
#     'kernel': ['rbf'],        # Most commonly effective kernels
#     'C': [0.1],                      # Reasonable regularization strengths for regression
#     'epsilon': [0.01],             # Standard values for error tolerance
#     'gamma': ['scale']             # Default and a specific small value for tuning influence
# }

hyperparameter_grid_SVM = {
    'kernel': ['rbf'],  # A non-linear and linear option
    'C': [0.1, 1],               # Two reasonable regularization strengths
    'epsilon': [0.1, 0.5],       # Moderate tolerances for prediction errors
    'gamma': ['scale', 'auto'],          # Default gamma for simplicity
}

hyperparameter_grid_ENR = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.2, 0.5, 0.8]
}

# hyperparameter_grid_XGboost = {
#     'n_estimators': [50, 100],          # Number of trees (lower values for quicker training)
#     'max_depth': [3, 5, 7],             # Maximum depth of each tree (shallower trees to avoid overfitting)
#     'learning_rate': [0.01, 0.1],       # Learning rate (smaller values for more gradual training)
#     'subsample': [0.6, 0.8],            # Subsample ratio of the training instance (to prevent overfitting)
#     'colsample_bytree': [0.6, 0.8],     # Subsample ratio of columns when constructing each tree
#     'gamma': [0, 0.1],                  # Minimum loss reduction required to make a further partition on a leaf node
# }




LSTM_master_ = base_path_ / Path(f'LSTM folder')





