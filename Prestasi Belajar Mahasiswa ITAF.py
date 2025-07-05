# 🎓 Prediksi Prestasi Akademik Mahasiswa Menggunakan Backpropagation Neural Network (BPNN)

Proyek ini membangun model klasifikasi prestasi mahasiswa (kategori: Tinggi, Sedang, Rendah) berdasarkan **data perilaku belajar**, menggunakan algoritma **Backpropagation Neural Network (BPNN)**.

---

## 📊 Dataset

Data diambil dari lingkungan kampus ITAF Kupang dan mencakup variabel:

- Kehadiran (%)
- Nilai Tugas
- Keterlambatan Pengumpulan Tugas
- Nilai UTS
- Nilai UAS

Target output:  
✅ **Kategori Prestasi Akademik** (Tinggi / Sedang / Rendah)

---

## 🤖 Metode

- Model: **Backpropagation Neural Network (Multilayer Perceptron)**
- Aktivasi: Sigmoid
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epoch: 200
- Split: 80% pelatihan, 20% pengujian

---

## ✅ Hasil

- **Akurasi**: 91.5%
- **Mean Square Error (MSE)**: 0.045
- **Evaluasi Confusion Matrix**:

|               | Pred: Rendah | Pred: Sedang | Pred: Tinggi |
|---------------|--------------|--------------|---------------|
| **Akt: Rendah** |      8       |      2       |       1       |
| **Akt: Sedang** |      1       |     15       |       2       |
| **Akt: Tinggi** |      0       |      1       |      19       |

- **Precision**:  
  - Rendah: 0.89  
  - Sedang: 0.83  
  - Tinggi: 0.90  
- **Recall**:  
  - Rendah: 0.73  
  - Sedang: 0.83  
  - Tinggi: 0.95

---

## 🛠️ Tools & Library

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas / Numpy / Matplotlib / Seaborn

---

## 🚀 Cara Menjalankan

1. Install library:
```bash
pip install -r requirements.txt
