import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Create a simulated target variable (pKi values)
np.random.seed(42)
pKi_values = np.concatenate([np.random.uniform(9, 10, 80), np.random.uniform(6, 9, 20)])
print(pKi_values)
# Create a DataFrame
df = pd.DataFrame({'pKi': pKi_values})

# Create bins for the stratification: one bin for 6-9, and one bin for 9-10
bins = [6, 9, 10]
labels = ['6-9', '9-10']
df['pKi_range'] = pd.cut(df['pKi'], bins=bins, labels=labels)

# Show the distribution before stratification
print("Distribution of pKi ranges:")
print(df['pKi_range'])
