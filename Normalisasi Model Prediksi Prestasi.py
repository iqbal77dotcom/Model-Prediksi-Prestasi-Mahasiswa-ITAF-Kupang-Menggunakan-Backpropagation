import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Contoh data mentah (misalnya sudah dibaca dari file Excel/CSV)
data = pd.DataFrame({
    'kehadiran': [80, 90, 75, 85, 70],
    'nilai_tugas': [75, 80, 70, 90, 65],
    'keterlambatan_tugas': [2, 1, 3, 0, 4],
    'nilai_uts': [78, 85, 69, 80, 55],
    'nilai_uas': [82, 90, 74, 88, 60]
})

# Normalisasi Min-Max
scaler = MinMaxScaler(feature_range=(0.1, 0.9))
normalized_data = scaler.fit_transform(data)

# Hasilkan DataFrame baru
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
print(normalized_df)
