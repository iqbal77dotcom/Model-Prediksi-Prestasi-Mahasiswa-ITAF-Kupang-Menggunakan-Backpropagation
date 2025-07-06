import pandas as pd
import numpy as np

# Prediksi probabilitas (softmax)
y_pred_probs = model.predict(X_test)

# Ambil label prediksi (0, 1, 2)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# Ubah ke nama kelas (Rendah, Sedang, Tinggi)
y_pred_names = le.inverse_transform(y_pred_labels)

# Hitung jumlah prediksi per kategori
prediksi_df = pd.DataFrame({'Prediksi': y_pred_names})
hasil_prediksi = prediksi_df['Prediksi'].value_counts().sort_index()

print("Hasil Prediksi per Kategori:\n", hasil_prediksi)
