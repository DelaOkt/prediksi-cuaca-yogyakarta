import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

# ======================= Judul =======================
st.set_page_config(layout="wide")
st.title("📊 Prediksi Curah Hujan Harian Yogyakarta Tahun 2024")
st.markdown("""
Dashboard ini menyajikan hasil prediksi curah hujan harian di Yogyakarta tahun 2024
menggunakan model Machine Learning LSTM dengan bobot ekstrem.  
Data diperoleh dari BMKG Stasiun Klimatologi Yogyakarta (2020–2024).
""")

# ======================= Load Data =======================
df_pred = pd.read_csv("prediksi_2024_weighted_mm.csv")
df_pred['TANGGAL'] = pd.to_datetime(df_pred['TANGGAL']).dt.date
df_pred['BULAN'] = pd.to_datetime(df_pred['TANGGAL']).dt.month

# ======================= 1️⃣ Grafik Harian Jan–Des =======================
st.subheader("1️⃣ Grafik Prediksi vs Aktual Tahun 2024 (LSTM Bobot Ekstrem)")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_pred['TANGGAL'], df_pred['AKTUAL_MM'], label='Aktual', marker='o', markersize=2)
ax.plot(df_pred['TANGGAL'], df_pred['PREDIKSI_MM'], label='Prediksi (LSTM Bobot Ekstrem)', linestyle='--', marker='x', markersize=2)
ax.set_xlabel("Bulan")
ax.set_ylabel("Curah Hujan (mm)")
ax.set_title("Prediksi vs Aktual Curah Hujan Harian Tahun 2024")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.xlim([datetime.date(2024, 1, 1), datetime.date(2024, 12, 31)])
plt.xticks(rotation=0)
st.pyplot(fig)

# ======================= 1B. Grafik Per Bulan =======================
st.subheader("📆 Grafik Per Bulan")
selected_month = st.selectbox("Pilih Bulan", options=range(1, 13), format_func=lambda x: datetime.date(1900, x, 1).strftime('%B'))
monthly_df = df_pred[df_pred['BULAN'] == selected_month]
if not monthly_df.empty:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(monthly_df['TANGGAL'], monthly_df['AKTUAL_MM'], label='Aktual', marker='o')
    ax2.plot(monthly_df['TANGGAL'], monthly_df['PREDIKSI_MM'], label='Prediksi', linestyle='--', marker='x')
    ax2.set_title(f"Curah Hujan Harian Bulan {datetime.date(1900, selected_month, 1).strftime('%B')}")
    ax2.set_ylabel("RR (mm)")
    ax2.set_xlabel("Tanggal")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("Tidak ada data untuk bulan ini.")

# ======================= 1C. Klasifikasi Hari Berdasarkan Curah Hujan =======================
st.subheader("⚠️ Klasifikasi Hari Berdasarkan Curah Hujan")
labels = ['☀️ Tidak Hujan', '🌦️ Hujan Ringan', '🌧️ Hujan Sedang', '⚠️ Cuaca Ekstrem']
df_pred['KLASIFIKASI'] = pd.cut(df_pred['PREDIKSI_MM'], bins=[-0.1, 0, 20, 50, float('inf')], labels=labels)
st.dataframe(df_pred[['TANGGAL', 'PREDIKSI_MM', 'KLASIFIKASI']])

# ======================= 1E. Grafik Jumlah Hari per Kategori Hujan per Bulan =======================
st.subheader("🌧️ Jumlah Hari per Kategori Curah Hujan (2024)")
kategori_per_bulan = df_pred.groupby(['BULAN', 'KLASIFIKASI']).size().unstack(fill_value=0)
fig4, ax4 = plt.subplots(figsize=(10, 5))
kategori_per_bulan.plot(kind='bar', stacked=True, ax=ax4, colormap='tab20')
ax4.set_title("Jumlah Hari Berdasarkan Kategori Curah Hujan per Bulan (2024)")
ax4.set_xlabel("Bulan")
ax4.set_ylabel("Jumlah Hari")
ax4.set_xticklabels([datetime.date(1900, m, 1).strftime('%b') for m in kategori_per_bulan.index], rotation=0)
ax4.legend(title="Kategori Hujan", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig4)

# ======================= 1D. Grafik Akumulasi Bulanan =======================
st.subheader("📊 Akumulasi Curah Hujan per Bulan")
df_bulanan = df_pred.groupby('BULAN').agg({'AKTUAL_MM': 'sum', 'PREDIKSI_MM': 'sum'})
fig3, ax3 = plt.subplots()
ax3.bar(df_bulanan.index - 0.2, df_bulanan['AKTUAL_MM'], width=0.4, label='Aktual')
ax3.bar(df_bulanan.index + 0.2, df_bulanan['PREDIKSI_MM'], width=0.4, label='Prediksi')
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels([datetime.date(1900, m, 1).strftime('%b') for m in range(1, 13)])
ax3.set_ylabel("Total RR (mm)")
ax3.set_title("Akumulasi Curah Hujan per Bulan")
ax3.legend()
st.pyplot(fig3)

# ======================= 2️⃣ Evaluasi Model =======================
st.subheader("2️⃣ Evaluasi Model (LSTM Bobot Ekstrem)")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", "0.3289")
col2.metric("MAPE", "339.09%")
col3.metric("R² Score", "0.2238")

# ======================= 3️⃣ Prediksi Harian Interaktif =======================
st.subheader("3️⃣ Prediksi Harian")
selected_date = st.date_input("Pilih tanggal di tahun 2024", datetime.date(2024, 6, 1))
pred_row = df_pred[df_pred['TANGGAL'] == selected_date]
if not pred_row.empty:
    pred_rr = pred_row['PREDIKSI_MM'].values[0]
    st.success(f"Prediksi RR pada {selected_date.strftime('%d-%b-%Y')} adalah {pred_rr:.2f} mm")
    if pred_rr >= 50:
        st.error("⚠️ Cuaca Ekstrem: Potensi hujan sangat lebat!")
    elif pred_rr >= 20:
        st.warning("🌧️ Hujan sedang hingga lebat.")
    elif pred_rr > 0:
        st.info("🌦️ Hujan ringan.")
    else:
        st.info("☀️ Tidak ada hujan.")
else:
    st.warning("Data tidak tersedia untuk tanggal tersebut.")

# ======================= 4️⃣ Download Prediksi =======================
st.subheader("4️⃣ Unduh Hasil Prediksi")
st.download_button(
    label="📥 Download CSV Prediksi 2024 (mm)",
    data=df_pred.to_csv(index=False).encode('utf-8'),
    file_name="prediksi_2024_weighted_mm.csv",
    mime='text/csv'
)

# ======================= 5️⃣ Klasifikasi Curah Hujan =======================
st.subheader("5️⃣ Klasifikasi Curah Hujan")
st.markdown("""
- ☀️ **Tidak Hujan**: 0 mm  
- 🌦️ **Hujan Ringan**: 0 - 20 mm  
- 🌧️ **Hujan Sedang - Lebat**: 20 - 49.9 mm  
- ⚠️ **Cuaca Ekstrem**: ≥ 50 mm
""")

# ======================= 6️⃣ Kesimpulan =======================
st.subheader("6️⃣ Kesimpulan")
st.markdown("""
Model terbaik adalah **LSTM dengan bobot ekstrem**, yang ditujukan khusus untuk meningkatkan sensitivitas terhadap kejadian hujan ekstrem.  
Prediksi menunjukkan pola musiman yang mendekati aktual dan dapat digunakan sebagai alat bantu deteksi dini cuaca ekstrem di wilayah Yogyakarta.
""")
