import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import public_variables
from scipy import stats
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

def calculate_molecular_weight_from_pdb(pdb_file):
    """
    Calculate the molecular weight of a molecule from a PDB file.
    
    Parameters:
    - pdb_file: str, path to the PDB file
    
    Returns:
    - molecular_weight: float, molecular weight of the molecule
    """
    # Read the molecule from the PDB file
    mol = Chem.MolFromPDBFile(pdb_file)
    if mol is not None:
        return Descriptors.ExactMolWt(mol)  # Use the correct import
    else:
        print(f"Could not read molecule from {pdb_file}.")
        return None

def calculate_molecular_weights(folder_path):
    """
    Calculate the molecular weights of all PDB files in a specified folder.
    
    Parameters:
    - folder_path: str, path to the folder containing PDB files
    
    Returns:
    - weights: list of tuples, each containing (filename_without_extension, molecular_weight)
    """
    weights = []
    for filename in os.listdir(folder_path):
        
        if filename.endswith('.pdb'):
            pdb_file_path = os.path.join(folder_path, filename)
            mol_weight = calculate_molecular_weight_from_pdb(pdb_file_path)
            if mol_weight is not None:
                # Create a tuple with the filename without the extension and the molecular weight as an integer
                weights.append((int(filename[:-4]), mol_weight))  # Remove the last 4 characters (.pdb)

    return weights

def combine_pki_and_weights(pki_values, molecular_weights):
    """
    Combine two lists of tuples based on matching mol_id to create a new tuple
    with PKI value and molecular weight.
    
    Parameters:
    - pki_values: list of tuples, where each tuple is (mol_id, PKI_value)
    - molecular_weights: list of tuples, where each tuple is (mol_id, molecular_weight)
    
    Returns:
    - combined: list of tuples, where each tuple is (mol_id, PKI_value, molecular_weight)
    """
    # Create a dictionary for molecular weights with mol_id as the key
    weight_dict = {mol_id: weight for mol_id, weight in molecular_weights}
    
    # Combine the lists based on matching mol_id
    combined = []
    for mol_id, pki in pki_values:
        if mol_id in weight_dict:
            combined.append((mol_id, pki, weight_dict[mol_id]))
    
    return combined

def plot_pki_vs_molecular_weight(combined_data):
    """
    Plot PKI values against molecular weights from the combined data.
    
    Parameters:
    - combined_data: list of tuples, where each tuple is (mol_id, PKI_value, molecular_weight)
    
    Returns:
    - None
    """
    # Extract PKI values and molecular weights into separate lists
    pki_values = [pki for _, pki, _ in combined_data]
    molecular_weights = [weight for _, _, weight in combined_data]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(pki_values,molecular_weights, color='blue', alpha=0.7)

    # Customize the plot
    plt.title('Molecular Weight vs PKI')
    plt.ylabel('Molecular Weight (g/mol)')
    plt.xlabel('PKI Value')
    plt.grid()
    
    # Show the plot
    plt.savefig('pki_vs_molecular_weight.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
csv_file_path = public_variables.dfs_descriptors_only_path_ / '1ns.csv'
df = pd.read_csv(csv_file_path)
molid_pki_tuples = list(zip(df['mol_id'], df['PKI']))
# print(molid_pki_tuples)

folder_path = public_variables.base_path_ / 'pdb_GSK3'
molecular_weights = calculate_molecular_weights(folder_path)

combined_data = combine_pki_and_weights(molid_pki_tuples, molecular_weights)
# print(combined_data)

plot_pki_vs_molecular_weight(combined_data)