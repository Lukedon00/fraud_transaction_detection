# preprocess.py
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(directory_path):
    # Combine all .pkl files from the directory into a single DataFrame
    all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.pkl')]
    
    df_combined = pd.concat([pd.read_pickle(file) for file in all_files])

    # Encoding and feature engineering
    label_encoder_customer = LabelEncoder()
    label_encoder_terminal = LabelEncoder()
    df_combined['CUSTOMER_ID'] = label_encoder_customer.fit_transform(df_combined['CUSTOMER_ID'])
    df_combined['TERMINAL_ID'] = label_encoder_terminal.fit_transform(df_combined['TERMINAL_ID'])

    # Convert categorical columns to 'category' dtype
    df_combined['CUSTOMER_ID'] = df_combined['CUSTOMER_ID'].astype('category')
    df_combined['TERMINAL_ID'] = df_combined['TERMINAL_ID'].astype('category')

    # Convert TX_DATETIME to useful numerical features (e.g., hour, day, timestamp)
    df_combined['TX_DATETIME'] = pd.to_datetime(df_combined['TX_DATETIME'])
    
    # Example: Extract hour of the transaction
    df_combined['TX_HOUR'] = df_combined['TX_DATETIME'].dt.hour
    
    # Example: Extract the number of days since the start of the dataset (you can adjust this)
    df_combined['TX_TIME_DAYS'] = (df_combined['TX_DATETIME'] - df_combined['TX_DATETIME'].min()).dt.days
    
    # Optionally, drop the original TX_DATETIME column if not needed anymore
    df_combined = df_combined.drop(columns=['TX_DATETIME'])
    
    # Scaling numerical features
    scaler = StandardScaler()
    df_combined[['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']] = scaler.fit_transform(df_combined[['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']])

    # Splitting the data
    X = df_combined.drop(columns=['TX_FRAUD', 'TX_FRAUD_SCENARIO'])
    y = df_combined['TX_FRAUD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test
