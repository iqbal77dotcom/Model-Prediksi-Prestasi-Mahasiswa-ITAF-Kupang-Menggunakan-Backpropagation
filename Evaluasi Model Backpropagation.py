from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
import numpy as np

# === 1. Prediksi data uji (hasil dari model softmax)
y_pred_probs = model.predict(X_test)

# === 2. Ambil kelas tertinggi dari hasil prediksi dan target (jika one-hot)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# === 3. Cek distribusi label
print("Distribusi Label Asli:", np.bincount(y_test_labels))
print("Distribusi Prediksi  :", np.bincount(y_pred_labels))

# === 4. Akurasi
acc = accuracy_score(y_test_labels, y_pred_labels)
print("\nAkurasi:", round(acc * 100, 2), "%")

# === 5. Confusion Matrix
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=[0, 1, 2])
print("\nConfusion Matrix:")
print(cm)

# === 6. Classification Report (hindari error jika hanya muncul 2 kelas)
class_names = ["Rendah", "Sedang", "Tinggi"]
print("\nClassification Report:")
print(classification_report(
    y_test_labels,
    y_pred_labels,
    labels=[0, 1, 2],
    target_names=class_names,
    zero_division=0
))

# === 7. MSE (antara probabilitas prediksi dan label one-hot)
mse = mean_squared_error(y_test, y_pred_probs)
print("\nMean Squared Error (MSE):", round(mse, 4))
