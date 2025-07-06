

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# === 1. Contoh Data Dummy (silakan ganti dengan file CSV asli)
data = pd.DataFrame({
    'kehadiran': [80, 70, 90, 85, 75],
    'nilai_tugas': [75, 70, 85, 80, 65],
    'keterlambatan': [10, 30, 5, 15, 40],
    'nilai_uts': [78, 65, 85, 80, 60],
    'nilai_uas': [82, 70, 90, 88, 62],
    'label': ['Sedang', 'Rendah', 'Tinggi', 'Tinggi', 'Rendah']
})

# === 2. Pisahkan fitur dan label
X = data[['kehadiran', 'nilai_tugas', 'keterlambatan', 'nilai_uts', 'nilai_uas']]
y = data['label']

# === 3. Normalisasi fitur (Min-Max 0.1–0.9)
scaler = MinMaxScaler(feature_range=(0.1, 0.9))
X_scaled = scaler.fit_transform(X)

# === 4. Label encoding dan one-hot encoding untuk label
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Hasil: 0, 1, 2
y_categorical = to_categorical(y_encoded)

# === 5. Split data (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# === 6. Bangun model BPNN
model = Sequential()
model.add(Dense(10, input_dim=5, activation='sigmoid'))  # Hidden layer
model.add(Dense(3, activation='softmax'))  # Output layer (3 kelas)

# === 7. Kompilasi model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# === 8. Latih model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)

# === 9. Tampilkan grafik akurasi pelatihan
plt.plot(history.history['accuracy'], label='Data Latih')
plt.plot(history.history['val_accuracy'], label='Data Uji')
plt.title('Grafik Akurasi Pelatihan Model BPNN')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)
plt.show()
