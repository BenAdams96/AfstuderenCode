import public_variables
import create_csv_dataframes
import correlation_matrices
import randomForest_MD_features_only_dic

import A_create_dataframes_folders
import A_add_MD_features_to_models
import A_randomForest_make_Models
import correlation_matrices_approach2
import correlation_matrices_approach3
import A_One_big_model

# def main():
#     print('Run big Main file')
#     #A_create_dataframes_folders.main() #create 'descriptors only' folder and put all the csv files in it
#     # combined_df_v = A_One_big_model.combine_csv_by_molecule_order_vertically(public_variables.dfs_descriptors_only_path_) #creates 'concat_ver.csv'
#     # combined_df_h = A_One_big_model.combine_csv_by_molecule_order_horizontally(public_variables.dfs_descriptors_only_path_) #creates 'concat_hor.csv'
#     print('create correlation matrices')
#     #correlation_matrices.main() #create 'reduced_t' folder and also the correlation matrices #NOTE: but not from concat_hor.csv and concat_ver.csv
#     # correlation_matrices_approach2.main()
#     #correlation_matrices_approach3.main()

#     # # add MD features to the reduced model
#     # print('add MD features to reduced model')
#     #A_add_MD_features_to_models.main() #NOTE: copies the reduced folder and adds the MD features to it. not of concat

#     # correlation_matrices.compute_and_visualize_correlation_matrices_dic(public_variables.dfs_reduced_and_MD_path_)
    
#     # print('only MD features')
#     randomForest_MD_features_only_dic.main()
#     # correlation_matrices.compute_and_visualize_correlation_matrices_dic(public_variables.dfs_MD_only_path_)

#     # A_randomForest_make_Models.main(public_variables.dfs_descriptors_only_path_)
#     # A_randomForest_make_Models.main(public_variables.dfs_reduced_path_)
#     # A_randomForest_make_Models.main(public_variables.dfs_reduced_and_MD_path_)
#     # A_randomForest_make_Models.main(public_variables.dfs_MD_only_path_)
    
#     return

def main():
    print('Run big Main file')
    A_create_dataframes_folders.main() #create 'descriptors only' folder and put all the csv files in it
    A_randomForest_make_Models.main(public_variables.dfs_descriptors_only_path_)

    combined_df_v = A_One_big_model.combine_csv_by_molecule_order_vertically(public_variables.dfs_descriptors_only_path_) #creates 'concat_ver.csv'
    combined_df_h = A_One_big_model.combine_csv_by_molecule_order_horizontally(public_variables.dfs_descriptors_only_path_) #creates 'concat_hor.csv'

    print('create correlation matrices')
    correlation_matrices.main() #create 'reduced_t' folder and also the correlation matrices #NOTE: but not from concat_hor.csv and concat_ver.csv
    A_randomForest_make_Models.main(public_variables.dfs_reduced_path_)
    # correlation_matrices_approach2.main()
    correlation_matrices_approach3.main()

    # # add MD features to the reduced model
    # print('add MD features to reduced model')
    A_add_MD_features_to_models.main() #NOTE: copies the reduced folder and adds the MD features to it. not of concat
    A_randomForest_make_Models.main(public_variables.dfs_reduced_and_MD_path_)
    # correlation_matrices.compute_and_visualize_correlation_matrices_dic(public_variables.dfs_reduced_and_MD_path_)
    
    # print('only MD features')
    randomForest_MD_features_only_dic.main()
    # correlation_matrices.compute_and_visualize_correlation_matrices_dic(public_variables.dfs_MD_only_path_)
    A_randomForest_make_Models.main(public_variables.dfs_MD_only_path_)
    A_One_big_model.main(public_variables.dfs_descriptors_only_path_)
    A_One_big_model.main(public_variables.dfs_reduced_path_)
    A_One_big_model.main(public_variables.dfs_reduced_and_MD_path_)
    A_One_big_model.main(public_variables.dfs_MD_only_path_)
    # A_randomForest_make_Models.main(public_variables.dfs_reduced_path_)
    # A_randomForest_make_Models.main(public_variables.dfs_reduced_and_MD_path_)
    # A_randomForest_make_Models.main(public_variables.dfs_MD_only_path_)
    
    return

main()