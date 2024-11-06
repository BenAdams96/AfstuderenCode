
import public_variables
import pandas as pd

def main():
    bigdf = pd.read_csv(public_variables.dataframes_master_ / 'conformations_1000.csv')
    # Sort by 'conformations (ns)' and then 'mol_id' to achieve the desired order
    bigdf = bigdf.sort_values(by=['conformations (ns)', 'mol_id']).reset_index(drop=True)
    
    # Save the sorted DataFrame if needed
    bigdf.to_csv(public_variables.dataframes_master_ / 'conformations_1000.csv', index=False)
    return

main()