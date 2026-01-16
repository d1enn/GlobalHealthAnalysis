import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kagglehub 
import warnings 
warnings.filterwarnings('ignore')

# set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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

# show basic infos
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\nBasic info:")
print(df.info())

# 1- show descriptive statistics
print("\nDescriptive statistics for numerical columns:")
print(df.describe())

# 2- check for the data types
print("\nData types:")
print(df.dtypes)

def validate_and_enhance_data(df):
    """ Validate the data quality and create epidemiological features"""
    print("="*60)
    print("DATA VALIDATION REPORT")
    print("="*60)

    df_clean = df.copy()

    # 1- validate data ranged
    print("\n1- Data Range Validation")

    #check the prevalence rates (should be 0-100)
    prevalence_range = (df['Prevalence Rate (%)'].min(), df['Prevalence Rate (%)'].max())
    print(f"  Prevalence Rate range: {prevalence_range[0]:.2f}% to {prevalence_range[1]:.2f}%")

    # check the mortality rates
    mortality_range = (df['Mortality Rate (%)'].min(), df['Mortality Rate (%)'].max())
    print(f"  Mortality Rate range: {mortality_range[0]:.2f}% to {mortality_range[1]:.2f}%")

    # check for any impossible values
    if df['Prevalence Rate (%)'].min() < 0 or df['Prevalence Rate (%)'].max() > 100:
        print(" Warning: Prevalence rates outside 0-100% range")

    # 2- epidemiology metrics
    print("\n2- Creating Epidemiological Metrics:")

    # fertality rate = mortality rate / incidence rate
    df_clean['Case_Fatality_Rate'] = df_clean['Mortality Rate (%)'] / df_clean['Incidence Rate (%)'].replace(0, np.nan)
    df_clean['Case_Fertality_Rate'] = df_clean['Case_Fatality_Rate'].fillna(0)
    print(f"  Created: Case_Fatality_Rate (CFR)")

    # healthcare system score
    healthcare_vars = ['Healthcare Access (%)', 'Doctors per 1000', 'Hospital Beds per 1000']
    df_clean['Healthcare_System_Score'] = df_clean[healthcare_vars].mean(axis=1)
    print(f"  Created: Healthcare_System_Score")
    
    # socioeconomic status score
    socio_vars = ['Per Capita Income (USD)', 'Education Index', 'Urbanisation Rate (%)']
    df_clean['SES_Score'] = df_clean[socio_vars].apply(lambda x: (x - x.min()) / (x.max() - x.min())).mean(axis=1)
    print(f"  Created: SES_Score")

    # disability adjusted life years per 100,000
    df_clean['DALYs_per_100k'] = df_clean['DALYs'] / df_clean['Population Affected'] * 100000
    print(f"  Created: DALYs_per_100k")

    # disease severity
    df_clean['Disease_Severity_Index'] = (
        df_clean['Mortality Rate (%)'] * 0.4 +
        df_clean['DALYs_per_100k'] * 0.3 +
        (100 - df_clean['Recovery Rate (%)']) * 0.3
    ) / 100
    print(f"  Created: Disease_Severity_Index")

    # Categorise diseases
    print("\n3- Disease Categorisation:")

    # create binary indicators for major disease categories
    disease_categories = df['Disease Category'].unique()
    print(f"  Found {len(disease_categories)} disease categories: {list(disease_categories)}")

    # create high burden flag (top 25% DALYs)
    dalys_threshold = df['DALYs'].quantile(0.75)
    df_clean['High_Burden_Disease'] = df_clean['DALYs'] > dalys_threshold
    print(f"  Created: High_Burden_Disease flag (threshold: {dalys_threshold:.0f} DALYs)")

    # summary stats
    print("\n4- Data Summary:")
    print(f"  Total observations: {len(df_clean)}")
    print(f"  Time period: {df_clean['Year'].min()} - {df_clean['Year'].max()}")
    print(f"  Countries: {df_clean['Country'].nunique()}")
    print(f"  Diseases: {df_clean['Disease Name'].nunique()}")

    return df_clean

df_enhanced = validate_and_enhance_data(df)