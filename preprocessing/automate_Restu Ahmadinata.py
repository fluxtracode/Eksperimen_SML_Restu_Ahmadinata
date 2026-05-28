import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def run_preprocessing():
    print("Memulai proses otomatisasi data Churn...")
    
    input_path = 'churn_raw.csv'
    output_dir = 'preprocessing'
    output_path = os.path.join(output_dir, 'churn_preprocessing.csv')
    
    # Memuat & Membersihkan
    df = pd.read_csv(input_path)
    df_clean = df.dropna().copy()
    if 'CustomerID' in df_clean.columns:
        df_clean = df_clean.drop('CustomerID', axis=1)
    
    # Encoding & Pemisahan Fitur
    df_encoded = pd.get_dummies(df_clean, columns=['Gender', 'Subscription Type', 'Contract Length'], drop_first=True)
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Normalisasi
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Menggabungkan dan Menyimpan
    df_preprocessed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_preprocessed.to_csv(output_path, index=False)
    print(f"Selesai. Data disimpan di: {output_path}")

if __name__ == "__main__":
    run_preprocessing()