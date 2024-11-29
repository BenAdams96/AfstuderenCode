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

import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
from typing import Dict
import numpy as np
from pathlib import Path
import joblib

import pandas as pd
import math
import re
import os

def main(dfs_path = public_variables.dfs_descriptors_only_path_):  ###set as default, but can always change it to something else.
    modelpath = dfs_path / "ModelResults_RF" / "original_models_dic.pkl"
    print(modelpath)
    with open(modelpath, "rb") as f:
        loaded_model_dic = pickle.load(f)

    
    return

if __name__ == "__main__":
    main(public_variables.dfs_descriptors_only_path_)


