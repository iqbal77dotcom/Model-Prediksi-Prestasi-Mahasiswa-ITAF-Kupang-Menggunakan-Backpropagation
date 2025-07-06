# 1. Lakukan prediksi terhadap semua data
y_all_probs = model.predict(X_scaled)  # hasil softmax probabilitas
y_all_labels = np.argmax(y_all_probs, axis=1)  # ambil label (0,1,2)
y_all_names = le.inverse_transform(y_all_labels)  # ubah ke: Rendah, Sedang, Tinggi

# 2. Buat dataframe hasil prediksi
hasil_df = data.copy()
hasil_df["Prediksi_Prestasi"] = y_all_names

# 3. Tampilkan rekap jumlah kategori
rekap_prediksi = hasil_df["Prediksi_Prestasi"].value_counts().sort_index()
print("Distribusi Prediksi Seluruh Mahasiswa (85 Data):\n")
print(rekap_prediksi)

# 4. (Opsional) Tampilkan 10 data teratas
print("\nContoh Hasil Prediksi:")
print(hasil_df[["kehadiran", "nilai_tugas", "keterlambatan", "nilai_uts", "nilai_uas", "Prediksi_Prestasi"]].head(10))
