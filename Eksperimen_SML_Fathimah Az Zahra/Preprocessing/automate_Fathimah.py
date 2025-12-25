import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import gdown
import zipfile
import os

# --- Fungsi 1: Ambil Data Mentah ---
def load_data():
    # Download dataset water potability dari GDrive
    file_id = '15afU0su_s4WMeX7nFKO5RiMQwsezW7KW'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'drinking_water.zip'

    # Bersihkan file lama jika ada
    if os.path.exists(output):
        os.remove(output)

    # Download (quiet=True biar ga berisik di log)
    gdown.download(url, output, quiet=True)

    # Ekstrak ZIP
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('dataset_air')

    # Baca CSV
    # Pastikan path sesuai hasil ekstrak
    df = pd.read_csv('dataset_air/water_potability.csv')
    return df

# --- Fungsi 2: Proses Data (Cleaning & Scaling) ---
def preprocess_data(df):
    # 1. Hapus Data Ganda
    df = df.drop_duplicates()

    # 2. Isi Data Kosong (Imputasi) dengan Rata-rata
    imputer = SimpleImputer(strategy='mean')
    # Imputer mengembalikan array, jadi kita balikin ke DataFrame
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # 3. Pisahkan Fitur (X) dan Target (y)
    X = df_imputed.drop('Potability', axis=1)
    y = df_imputed['Potability']

    # 4. Bagi Data Training (80%) dan Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Standarisasi (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Rapikan kembali ke bentuk DataFrame
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['Potability'] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['Potability'] = y_test.reset_index(drop=True)

    return train_df, test_df

# --- Main Program (Yang Dijalankan Robot) ---
if __name__ == "__main__":
    print("🤖 Memulai proses otomatisasi data...")

    # Langkah 1: Load
    df = load_data()
    print(f"✅ Data berhasil dimuat! Total baris: {len(df)}")

    # Langkah 2: Preprocess
    train_data, test_data = preprocess_data(df)
    print("✅ Data berhasil dibersihkan dan distandarisasi.")

    # Langkah 3: Simpan Hasil
    train_data.to_csv('water_potability_train_clean.csv', index=False)
    test_data.to_csv('water_potability_test_clean.csv', index=False)

    print("🎉 Selesai! File 'water_potability_train_clean.csv' siap digunakan.")
