import pandas as pd
from sklearn.preprocessing import StandardScaler

def pipeline_preprocessing_otomatis(file_path):
    """
    Fungsi otomatisasi preprocessing data Credit Risk.
    Menerima input path file CSV dan mengembalikan DataFrame yang siap dilatih.
    """
    # 1. Load Dataset
    df = pd.read_csv(file_path)

    df_clean = df.copy()
    
    # 2. Pembersihan Outlier
    # Batasi usia maksimal 80 tahun
    df_clean = df_clean[df_clean['person_age'] <= 80]
    # Batasi masa kerja maksimal 60 tahun
    df_clean = df_clean[df_clean['person_emp_length'] <= 60]
    # Batasi riwayat kredit maksimal 40 tahun
    df_clean = df_clean[df_clean['cb_person_cred_hist_length'] <= 40]
    
    # 3. Penanganan Missing Values (Imputasi Median)
    median_emp = df_clean['person_emp_length'].median()
    median_rate = df_clean['loan_int_rate'].median()
    
    df_clean['person_emp_length'] = df_clean['person_emp_length'].fillna(median_emp)
    df_clean['loan_int_rate'] = df_clean['loan_int_rate'].fillna(median_rate)
    
    # 4. Categorical Encoding (One-Hot Encoding)
    kolom_kategorikal = [
        'person_home_ownership', 
        'loan_intent', 
        'loan_grade', 
        'cb_person_default_on_file'
    ]

    df_clean = pd.get_dummies(df_clean, columns=kolom_kategorikal, drop_first=True)
    
    # 5. Feature Scaling (Standard Scaler)
    # Pisahkan kolom target (loan_status)
    target = 'loan_status'
    fitur_numerik = [
        'person_age', 'person_income', 'person_emp_length', 
        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
        'cb_person_cred_hist_length'
    ]
    
    scaler = StandardScaler()
    df_clean[fitur_numerik] = scaler.fit_transform(df_clean[fitur_numerik])
    
    print("-> Preprocessing selesai! Data siap digunakan.")
    return df_clean

# KODE DI BAWAH INI AKAN BERJALAN OTOMATIS JIKA FILE DI-RUN LANGSUNG
if __name__ == "__main__":
    # Ganti dengan nama file dataset asli Anda
    INPUT_DATASET = "credit_risk_dataset.csv" 
    OUTPUT_DATASET = "preprocessing/credit_risk_dataset_preprocessing.csv"
    
    try:
        # Jalankan fungsi otomatisasi
        data_siap_pakai = pipeline_preprocessing_otomatis(INPUT_DATASET)
        
        # Simpan hasilnya ke CSV baru untuk proses modeling nanti
        data_siap_pakai.to_csv(OUTPUT_DATASET, index=False)
        print(f"Sukses! File bersih disimpan dengan nama: {OUTPUT_DATASET}")
        
    except FileNotFoundError:
        print(f"Error: File '{INPUT_DATASET}' tidak ditemukan. Pastikan posisinya satu folder dengan script ini.")
