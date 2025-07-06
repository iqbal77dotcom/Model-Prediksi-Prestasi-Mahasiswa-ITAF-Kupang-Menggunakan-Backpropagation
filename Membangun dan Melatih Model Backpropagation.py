import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data (asumsikan sudah dibaca ke dalam DataFrame)
data = pd.read_csv("data_mahasiswa.csv")  # contoh: kehadiran, tugas, keterlambatan, UTS, UAS, label

# 2. Pisahkan fitur dan label
X = data[['kehadiran', 'nilai_tugas', 'keterlambatan', 'nilai_uts', 'nilai_uas']]
y = data['label']  # label: Rendah, Sedang, Tinggi

# 3. Normalisasi dengan MinMaxScaler (rentang 0.1–0.9)
scaler = MinMaxScaler(feature_range=(0.1, 0.9))
X_scaled = scaler.fit_transform(X)

# 4. Split data menjadi data latih dan data uji (80:20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Bangun dan latih model MLPClassifier (Backpropagation)
model = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd',
                      learning_rate_init=0.3, max_iter=1000, random_state=42)

model.fit(X_train, y_train)

# 6. Evaluasi awal terhadap data training dan data uji
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Akurasi Data Latih:", accuracy_score(y_train, train_preds))
print("Akurasi Data Uji:", accuracy_score(y_test, test_preds))
