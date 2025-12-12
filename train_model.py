import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. LOAD DATASET
try:
    df = pd.read_csv('datasets.csv', header=3)
except FileNotFoundError:
    print("Error: File 'datasets.csv' tidak ditemukan.")
    exit()

# 2. DATA CLEANING
# Hapus kolom yang tidak dipakai
drop_cols = ['NIM', 'Nama', 'SemesterDropout', 'JenisKelamin']
# Filter agar hanya menghapus kolom yang benar-benar ada (menghindari error)
existing_cols = [col for col in drop_cols if col in df.columns]
df_clean = df.drop(columns=existing_cols)

# Pisahkan Fitur (X) dan Target (y)
X = df_clean.drop(columns=['Target'])
y = df_clean['Target']

# 3. ENCODING
label_encoders = {}
cat_cols = X.select_dtypes(include=['object']).columns

print("Sedang memproses data kategori...")
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode Target
target_le = LabelEncoder()
y = target_le.fit_transform(y)

# 4. SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. SPLITTING
# Stratify=y wajib untuk data kecil agar semua kelas terwakili
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. TRAIN KNN (K=3 karena data sedikit)
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

# 7. EVALUASI
y_pred = knn.predict(X_test)
print(f"\nAkurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Fix Error Classification Report (Memaksa semua label muncul)
unique_labels = np.arange(len(target_le.classes_))
print("\nLaporan Klasifikasi:\n", classification_report(
    y_test, y_pred, labels=unique_labels, target_names=target_le.classes_
))

# 8. SIMPAN
artifacts = {
    'model': knn,
    'scaler': scaler,
    'encoders': label_encoders,
    'target_encoder': target_le,
    'feature_names': X.columns.tolist(),
    'cat_cols': cat_cols.tolist()
}
joblib.dump(artifacts, 'student_prediction_artifacts.pkl')