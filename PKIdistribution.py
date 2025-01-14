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
    print(csv_file)
    # Extract the second column (assuming it's the second column by index)
    # column_values = df.loc[:,'exp_mean [nM]']
    df['pKi'] = -np.log10(df['exp_mean [nM]'] * 1e-9)

    # # Filter values between 5 and 11
    # filtered_values = column_values[(column_values >= 5) & (column_values <= 11)]
    num_bins = 30
    # Calculate histogram
    # counts, bin_edges = np.histogram(df['pKi'], bins=np.arange(5, 11 + bin_size, bin_size))
    bin_edges = np.linspace(df['pKi'].min(),df['pKi'].nlargest(2).iloc[-1], num_bins+1)
    counts, bin_edges = np.histogram(df['pKi'], bins=bin_edges)
    mode_index = np.argmax(counts)  # Index of the maximum count
    mode_value = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2  # Midpoint of the bin with the highest count
    # Calculate the modal value from the histogram
    # mode_index = np.argmax(counts)  # Index of the maximum count
    # mode_value = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2  # Midpoint of the bin with the highest count

    # Calculate median and average
    median_value = np.median(df['pKi'])
    average_value = np.mean(df['pKi'])

    # Plot the distribution using a histogram with the specified bin size
    plt.figure(figsize=(10, 6))
    sns.histplot(df['pKi'], bins=bin_edges, kde=True)

    # Customize the plot
    plt.axvline(mode_value, color='blue', linestyle='--', label=f'Mode: {mode_value:.2f}')
    plt.axvline(median_value, color='orange', linestyle='--', label=f'Median: {median_value:.2f}')
    plt.axvline(average_value, color='green', linestyle='--', label=f'Average: {average_value:.2f}')
    
    plt.title('Distribution of pKi Values in GSK3 Dataset (856 Molecules)')
    plt.xlabel('PKI Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig('distribution_plot_GSK3.png', dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()


# Example usage
csv_file_path = public_variables.dataset_csvfile_path_
plot_column_distribution(csv_file_path, bin_size=0.1885) #0.188 looks awefully close to ppim