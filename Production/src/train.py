
import os
import joblib
import pandas as pd

try:
    
    from src.data_loader import load_data
    from src.pipeline import build_pipeline
except ModuleNotFoundError:
    from data_loader import load_data
    from pipeline import build_pipeline

# ---------------------------------------------------------
# 1) Balance the dataset
# ---------------------------------------------------------
def balance_dataset(df):
    """
    Downsample majority class to match minority.
    """
    target_col = 'HadDepressiveDisorder'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    df_majority = df[df[target_col] == 0]
    df_minority = df[df[target_col] == 1]

    ratio = 1
    majority_needed = len(df_minority) * ratio

    df_majority_sampled = df_majority.sample(n=majority_needed, random_state=42)

    df_balanced = pd.concat([df_majority_sampled, df_minority], axis=0)
    
    # (Shuffle)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced

# ---------------------------------------------------------
# 2) Train pipeline logic
# ---------------------------------------------------------
def train_pipeline(data_path, save_path="model/pipeline.pkl"):
    
    print(f"1. Loading Data from {data_path}...")
    
    df = load_data(data_path)

    print("2. Balancing Dataset...")
    df_balanced = balance_dataset(df)

    y = df_balanced['HadDepressiveDisorder']

    drop_list = [
        'State', 'LastCheckupTime', 'RemovedTeeth', 'HadAngina',
        'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
        'HadKidneyDisease', 'HadArthritis', 'ChestScan', 'BMI',
        'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
        'TetanusLast10Tdap', 'HadDepressiveDisorder', 
        'HadHeartAttack', 'HighRiskLastYear', 'SmokerStatus',
        'ECigaretteUsage'
    ]

    print("3. Preparing Features (X)...")
    X = df_balanced.drop(columns=drop_list, errors='ignore') 

    print(f"   - Features Count: {X.shape[1]}")
    print(f"   - Rows Count: {X.shape[0]}")

    print("4. Building Pipeline...")
    pipeline = build_pipeline()

    # Fit
    print("5. Training Model (CatBoost)...")
    pipeline.fit(X, y)

    # Saving
    print("6. Saving Model...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    
    print(f"✅ Success! Pipeline saved to: {save_path}")
    return pipeline

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    DATA_FILE_PATH = "K:\DEPI\Final Project\Mental_Model1\Mental_Model\Mental_Model\data\proccesed\Final.csv" 
    
    if os.path.exists(DATA_FILE_PATH):
        train_pipeline(DATA_FILE_PATH)
    else:
        print(f"❌ Error: Data file not found at {DATA_FILE_PATH}")