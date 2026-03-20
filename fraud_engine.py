import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save_model(data_path):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # 1. Filter to high-risk transactions
    relevant_types = ['TRANSFER', 'CASH_OUT']
    df = df[df['type'].isin(relevant_types)].copy()

    # 2. PM Feature Engineering (The Secret Sauce)
    print("Engineering risk features...")
    df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

    # NEW: Balance Drain Ratio to make the model sensitive to the percentage being sent
    # We use np.where to safely handle cases where the old balance is 0 to avoid division by zero errors
    df['balance_drain_ratio'] = np.where(df['oldbalanceOrg'] == 0, 0, df['amount'] / df['oldbalanceOrg'])

    # 3. Clean and Encode
    df_final = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    df_final = pd.get_dummies(df_final, columns=['type'], drop_first=True)

    X = df_final.drop(['isFraud'], axis=1)
    y = df_final['isFraud']

    # 4. Train the Model with High Sensitivity Class Weights
    print("Training the Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        class_weight={0: 1, 1: 100}, # Aggressive weighting
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # 5. Export Model and Column Layout
    joblib.dump(model, 'fraud_model.pkl')
    joblib.dump(X.columns.tolist(), 'model_columns.pkl')
    print("Success! 'fraud_model.pkl' and 'model_columns.pkl' saved.")

if __name__ == "__main__":
    # Make sure this matches the exact name of your CSV file
    train_and_save_model('PS_20174392719_1491204439457_log.csv')