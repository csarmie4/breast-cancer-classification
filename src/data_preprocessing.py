# src/data_processing.py

import pandas as pd

def load_data(file_path):
    """Load data from CSV."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data by converting target variable and dropping unnecessary features."""
    data = {'M': 1, 'B': 0}
    df['diagnosis'] = [data[item] for item in df['diagnosis']]
    df['diagnosis'] = df['diagnosis'].astype('category')
    
    # Drop features with '_se' and '_worst'
    df = df[df.columns.drop(list(df.filter(regex='worst')))]
    df = df[df.columns.drop(list(df.filter(regex='se')))]
    
    # Drop unnecessary features
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    print(df.head())
    return df
