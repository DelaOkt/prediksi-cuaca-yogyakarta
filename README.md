# prediksi-cuaca-yogyakarta

Sistem prediksi curah hujan berbasis web menggunakan Streamlit dengan algoritma XGBoost Tuned dan data historis dari BMKG Yogyakarta. Sistem ini dikembangkan dalam rangka penyusunan Tugas Akhir dan dapat melakukan:
1. Preprocessing data cuaca (pembersihan, penanganan missing values, normalisasi, deteksi outlier)
2. Prediksi curah hujan berdasarkan input manual atau file data
3. Evaluasi performa model (RMSE, MAPE, R²)
4. Deteksi cuaca ekstrem menggunakan threshold kuartil ketiga (Q3)
5. Visualisasi hasil dalam bentuk tabel dan grafik interaktif
6. Ekspor hasil ke Excel
