import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mn
import kagglehub 
import os

# download the dataset using kagglehub
path = kagglehub.dataset_download("malaiarasugraj/global-health-statistics")
print("Path to dataset files:", path)

# list all the files in the downloaded dir

print("\nFiles in dataset directory:")
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        print(f" - {filename} (full path: {file_path})")

# load the dataset
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"\nCSV files found: {csv_files}")

if csv_files:
    if len(csv_files) == 1:
        file_path = os.path.join(path, csv_files[0])

# load dataset
df = pd.read_csv(file_path)

# show basic info
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\nBasic info:")
print(df.info())

# check for any missing values
print(f"\nMissing values per column:")
print(df.isnull().sum())

#visualise the missing values
plt.figure(figsize=(12, 6))
mn.matrix(df)
plt.title('Missing Values Matric')
plt.show()

