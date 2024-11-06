
import pickle
from pathlib import Path
import public_variables

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_in_model_dictionary(models_path, pickle_file_name):
    with open(models_path / pickle_file_name, 'rb') as file:
        loaded_models_dict = pickle.load(file)
    return loaded_models_dict

def save_model_dictionary(models_path, pickle_filename, model_dictionary):
    with open(models_path / pickle_filename, 'wb') as file:
            pickle.dump(model_dictionary, file)
    return

def main():
    folder = public_variables.dfs_descriptors_only_path_
    base_path = Path(__file__).resolve().parent
    
    dfs_path = Path(base_path) / folder
    models_path = Path(base_path) / public_variables.Modelresults_RF / 'original models'
    
    #model_dictionary = {'RF_Allmodel_k5_RMSE.pkl': {'original models': [11 models] }, 'RF_reduced_Allmodels_k5_RMSE.pkl': {'original models': [11 models] }}
    model_dictionary = read_in_model_dictionary(models_path)
    
if __name__ == "__main__":
    
    main()







