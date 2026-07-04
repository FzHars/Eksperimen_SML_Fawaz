import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_dir):
    """
    Fungsi otomatisasi preprocessing data sesuai eksperimen di notebook.
    """
    # 1. Memuat Dataset
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File mentah tidak ditemukan di: {input_path}")
        
    df = pd.read_csv(input_path)
    
    # 2. Preprocessing: Menghapus data duplikat
    df = df.drop_duplicates()
    
    # 3. Memisahkan Fitur dan Target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Standardisasi Fitur (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Konversi kembali ke DataFrame agar mudah disimpan ke CSV
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # 6. Membuat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # 7. Menyimpan hasil preprocessing ke folder tujuan
    X_train_df.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    X_test_df.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"✓ Preprocessing selesai! File disimpan di folder: {output_dir}")

if __name__ == "__main__":
    # Path disesuaikan dengan struktur folder Dicoding
    INPUT_FILE = "../heart_disease_raw/heart.csv"
    OUTPUT_FOLDER = "heart_disease_preprocessing"
    
    preprocess_data(INPUT_FILE, OUTPUT_FOLDER)
