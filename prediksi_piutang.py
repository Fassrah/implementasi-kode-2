import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. DATA PREPARATION (Simulasi PT Maju Bersama)
# ==========================================
# Membuat 1000 data dummy invoice pelanggan
np.random.seed(42)
data = {
    'Total_Tagihan': np.random.randint(1000, 50000, 1000),
    'Limit_Hari_Kredit': np.random.choice([15, 30, 45, 60], 1000),
    'Metode_Bayar': np.random.choice(['Transfer Bank', 'Kartu Kredit', 'Tunai/Cek'], 1000),
    'Jml_Telat_Sebelumnya': np.random.randint(0, 5, 1000),
    'Status_Target': np.random.choice([0, 1], 1000, p=[0.85, 0.15]) # 1 = Telat Bayar, 0 = Tepat Waktu
}
df = pd.DataFrame(data)

# Merubah kolom text/kategorikal ('Metode_Bayar') menjadi numerik agar bisa dibaca mesin
df_encoded = pd.get_dummies(df, columns=['Metode_Bayar'], drop_first=True)

# ==========================================
# 2. DEFINISI FITUR (X) DAN TARGET (y)
# ==========================================
X = df_encoded.drop('Status_Target', axis=1) # Fitur yang dipelajari mesin
y = df_encoded['Status_Target']              # Jawaban yang harus ditebak mesin

# Membagi data untuk belajar (80%) dan untuk ujian (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. MODELING (Melatih Model Random Forest)
# ==========================================
# Menambahkan class_weight='balanced' karena data telat bayar lebih sedikit dibanding yg tepat waktu
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_rf.fit(X_train, y_train)

# ==========================================
# 4. EVALUASI (Ujian Hasil Prediksi)
# ==========================================
prediksi = model_rf.predict(X_test)

print("=== HASIL PREDIKSI KETERLAMBATAN INVOICE PT MAJU BERSAMA ===")
print(f"Akurasi Model: {accuracy_score(y_test, prediksi) * 100:.2f}%")
print("\nLaporan Detail (Classification Report):")
print(classification_report(y_test, prediksi, target_names=['Tepat Waktu (0)', 'Telat Bayar (1)']))

# Menampilkan fitur mana yang paling penting/berpengaruh membuat orang telat bayar
importances = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFaktor Paling Berpengaruh terhadap Keterlambatan:")
print(importances)