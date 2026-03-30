import pandas as pd
import numpy as np

def handle_rate(value):
    """
    Cleans the rate column by removing the '/5' denominator and converting to float.
    """
    try:
        value = str(value).split('/')
        value = value[0]
        return float(value)
    except (ValueError, IndexError):
        return np.nan

def bin_price(price):
    """
    Categorizes the approximate cost into Budget, Mid-range, and Luxury.
    """
    if price < 500:
        return 'Budget'
    elif 500 <= price < 1000:
        return 'Mid-range'
    else:
        return 'Luxury'

def detect_outliers_iqr(df, column):
    """
    Detects outliers using the Interquartile Range (IQR) method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def encode_categorical(df):
    """
    Performs industry-standard encoding:
    - Label Encoding for Binary/Ordinal features.
    - One-Hot Encoding for Nominal features.
    """
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = df.copy()
    le = LabelEncoder()
    
    # Label Encoding for Binary columns
    for col in ['online_order', 'book_table']:
        if col in df_encoded.columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
    # Ordinal Encoding for price_range
    price_map = {'Budget': 0, 'Mid-range': 1, 'Luxury': 2}
    if 'price_range' in df_encoded.columns:
        df_encoded['price_range'] = df_encoded['price_range'].map(price_map)
        
    # One-Hot Encoding for Nominal column: listed_in(type)
    if 'listed_in(type)' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['listed_in(type)'], prefix='type', drop_first=True)
        
    return df_encoded
