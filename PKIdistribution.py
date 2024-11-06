import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import public_variables
from scipy import stats
import numpy as np

def plot_column_distribution(csv_file, bin_size=0.1):
    """
    Reads a CSV file, extracts the second column, filters values between 5 and 11,
    and plots a bar plot of their distribution along with modal, median, and average values.
    
    Parameters:
    - csv_file: str, path to the CSV file
    - bin_size: float, size of the bins for the bar plot
    
    Returns:
    - None
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract the second column (assuming it's the second column by index)
    column_values = df.iloc[:, 1]

    # Filter values between 5 and 11
    filtered_values = column_values[(column_values >= 5) & (column_values <= 11)]

    # Calculate histogram
    counts, bin_edges = np.histogram(filtered_values, bins=np.arange(5, 11 + bin_size, bin_size))

    # Calculate the modal value from the histogram
    mode_index = np.argmax(counts)  # Index of the maximum count
    mode_value = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2  # Midpoint of the bin with the highest count

    # Calculate median and average
    median_value = np.median(filtered_values)
    average_value = np.mean(filtered_values)

    # Plot the distribution using a histogram with the specified bin size
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_values, bins=np.arange(5, 11 + bin_size, bin_size), kde=False)

    # Customize the plot
    plt.axvline(mode_value, color='blue', linestyle='--', label=f'Mode: {mode_value:.2f}')
    plt.axvline(median_value, color='orange', linestyle='--', label=f'Median: {median_value:.2f}')
    plt.axvline(average_value, color='green', linestyle='--', label=f'Average: {average_value:.2f}')
    
    plt.title('Distribution of PKI Values in JAK1 Dataset (567 Molecules)')
    plt.xlabel('PKI Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig('distribution_plot.png', dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()


# Example usage
csv_file_path = public_variables.dfs_descriptors_only_path_ / '1ns.csv'
plot_column_distribution(csv_file_path, bin_size=0.2)