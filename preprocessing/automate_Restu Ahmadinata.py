import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def run_preprocessing():
    print("Memulai proses otomatisasi data Churn...")
    
    input_path = 'churn_raw.csv'
    output_dir = 'preprocessing'
    output_path = os.path.join(output_dir, 'churn_preprocessing.csv')
    
    try:
        # 1. Memuat Data
        print(f"Membaca data dari {input_path}...")
        df = pd.read_csv(input_path)
        
        # 2. Membersihkan Data & Drop CustomerID
        df_clean = df.dropna().copy()
        if 'CustomerID' in df_clean.columns:
            df_clean = df_clean.drop('CustomerID', axis=1)
        
        # 3. Encoding Data Kategorikal (Ubah teks jadi angka 0 dan 1)
        print("Melakukan encoding pada kolom kategorikal...")
        df_encoded = pd.get_dummies(
            df_clean, 
            columns=['Gender', 'Subscription Type', 'Contract Length'], 
            drop_first=True,
            dtype=int
        )
        
        # 4. Memisahkan fitur (X) dan target (y)
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
        
        # 5. Normalisasi
        print("Normalisasi data menggunakan MinMaxScaler...")
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # 6. Menggabungkan dan Menyimpan
        df_preprocessed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df_preprocessed.to_csv(output_path, index=False)
        print(f"Selesai! Data berhasil diproses dan disimpan di: {output_path}")
        
    except Exception as e:
        print(f"Terjadi kesalahan saat memproses data: {e}")

if __name__ == "__main__":
    run_preprocessing()

# end of the line (1)