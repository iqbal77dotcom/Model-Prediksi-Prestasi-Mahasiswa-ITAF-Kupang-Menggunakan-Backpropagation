

# Import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
data = pd.read_excel('data_prestasi_mahasiswa.xlsx')  # Ganti dengan path file kamu

# Tampilkan 5 data teratas
print(data.head())

# Preprocessing
# Pisahkan fitur (X) dan target (y)
X = data[['Kehadiran', 'Nilai_Tugas', 'Keterlambatan_Tugas', 'Nilai_UTS', 'Nilai_UAS']]
y = data['Kategori']  # Target: 'Tinggi', 'Sedang', 'Rendah'

# Label encoding target
y = y.map({'Rendah': 0, 'Sedang': 1, 'Tinggi': 2})

# Normalisasi data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Bangun model BPNN
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))  # 3 kelas: Rendah, Sedang, Tinggi

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Latih model
history = model.fit(X_train, y_train, epochs=200, batch_size=8, validation_data=(X_test, y_test))

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Akurasi: {accuracy*100:.2f}%')

# Prediksi
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rendah', 'Sedang', 'Tinggi'], yticklabels=['Rendah', 'Sedang', 'Tinggi'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

# Laporan klasifikasi
print(classification_report(y_test, y_pred_classes, target_names=['Rendah', 'Sedang', 'Tinggi']))
