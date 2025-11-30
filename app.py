import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Laptop Price Predictor", layout="wide", page_icon="💻")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model(path="model_pipeline_NEW.joblib"):
    path = Path(path)
    if not path.exists():
        st.error(f"❌ Model file tidak ditemukan: {path.resolve()}")
        return None
    return joblib.load(path)

model = load_model("model_pipeline_NEW.joblib")

# -------------------------
# Options
# -------------------------
company_options = [
    "Acer","Apple","Asus","Chuwi","Dell","Fujitsu","Google","HP","Huawei","LG",
    "Lenovo","Mediacom","Microsoft","MSI","Razer","Samsung","Toshiba","Vero","Xiaomi"
]
cpu_company_options = ["AMD","Intel","Samsung"]
cpu_type_options = [
    "AMD A-Series","AMD E-Series","Atom","Celeron","Core i3","Core i5","Core i7",
    "Pentium","Ryzen","Other"
]
gpu_company_options = ["AMD","ARM","Intel","Nvidia"]
storage_type_options = ["Flash","Flash+HDD","HDD","Hybrid","SSD","SSD+HDD","SSD+Hybrid"]
memory_unit_options = ['MB','GB','TB']
opsys_options = ['Mac','No OS','Windows','Linux/Other']
resolution_options = ["4K","FullHD","HD","QHD","Other"]

# -------------------------
# GPU tier function
# -------------------------
def map_gpu_to_tier(gpu_name):
    s = str(gpu_name).lower()
    if any(kw in s for kw in ['rtx','gtx','geforce','quadro','firepro','radeon rx']):
        return 2  # High
    if any(mid_kw in s for mid_kw in ['mx','940','930','920','540','530','520','r7','r5','mali']):
        return 1  # Mid
    if any(low_kw in s for low_kw in ['hd graphics','uhd graphics','iris','intel hd','intel uhd']):
        return 0  # Low
    return 1  # Default mid

gpu_perf_options = [
    "NVIDIA GeForce RTX 4090","NVIDIA GeForce RTX 3080","NVIDIA GeForce GTX 1080",
    "AMD Radeon RX 6800M","NVIDIA Quadro","AMD FirePro",
    "NVIDIA GeForce MX450","NVIDIA GeForce 940MX","AMD Radeon R7",
    "Intel Iris Xe","Intel UHD Graphics 620",
    "Intel HD Graphics 4000","Intel UHD Graphics 605","AMD Radeon R5",
    "ARM Mali-G52","Intel Iris Plus","Other"
]

# -------------------------
# UI Title
# -------------------------
st.markdown("""
<h1 style='text-align:center; color:#1E90FF;'>💻 Laptop Price Predictor</h1>
<p style='text-align:center; color:gray;'>Masukkan spesifikasi laptop dan lihat estimasi harganya + tren prediksi user lain</p>
<hr>
""", unsafe_allow_html=True)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# Sisa kode Streamlit Anda
st.title("Aplikasi dengan Background dari File CSS")
# -------------------------
# Input Form
# -------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("🏢 Brand (Company)", company_options)
        cpu_brand = st.selectbox("🧠 Brand CPU", cpu_company_options)
        cpu_type = st.selectbox("⚙️ Tipe CPU", cpu_type_options)
        cpu_freq = st.number_input("⚡ Frekuensi CPU (GHz)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
        gpu_brand = st.selectbox("🎮 Brand GPU", gpu_company_options)
        gpu_perf = st.selectbox("🔻 Model GPU", gpu_perf_options, index=2)
        gpu_tier = map_gpu_to_tier(gpu_perf)
        gpu_level = {0: "Low", 1: "Mid", 2: "High"}[gpu_tier]
        st.info(f"🧠 GPU Performance Level: **{gpu_level}**")
        storage_type = st.selectbox("💾 Jenis Storage", storage_type_options)
        memory_value = st.number_input("💽 Kapasitas Storage (angka)", min_value=0.0, value=256.0)
        memory_unit = st.selectbox("📐 Satuan Storage", memory_unit_options, index=1)

    with col2:
        ram_gb = st.number_input("🔧 RAM (GB)", min_value=1, max_value=128, value=8, step=1)
        opsys = st.selectbox("🖥️ Sistem Operasi", opsys_options)
        weight = st.number_input("⚖️ Berat (kg)", min_value=0.0, max_value=9.0, value=2.0, step=0.1)
        inches = st.number_input("📏 Ukuran Layar (Inci)", min_value=0.0, max_value=100.0, value=15.6, step=0.1)
        resolution = st.selectbox("🖼️ Resolusi Layar", resolution_options)
        ppi = st.number_input("🔍 PPI (Pixels Per Inch)", min_value=0.0, max_value=1000.0, value=141.0, step=1.0)
        ips_choice = st.selectbox("🌈 Layar IPS", ["Yes","No"], index=1)
        touchscreen_choice = st.selectbox("🖐️ Touchscreen", ["Yes","No"], index=1)

    submitted = st.form_submit_button("🚀 Predict Price")

# -------------------------
# When submitted
# -------------------------
if submitted:
    if model is None:
        st.stop()

    if memory_unit == "TB":
        mem_tb = float(memory_value)
    elif memory_unit == "GB":
        mem_tb = float(memory_value) / 1024.0
    elif memory_unit == "MB":
        mem_tb = float(memory_value) / (1024.0**2)
    else:
        mem_tb = float(memory_value) / 1024.0

    ips = 1 if ips_choice == "Yes" else 0
    touchscreen = 1 if touchscreen_choice == "Yes" else 0

    data_input = pd.DataFrame({
        'Inches':[inches],
        'CPU_Frequency (GHz)':[cpu_freq],
        'RAM (GB)':[ram_gb],
        'Weight (kg)':[weight],
        'Touchscreen':[touchscreen],
        'IPS':[ips],
        'PPI':[ppi],
        'Memory_TB':[mem_tb],
        'GPU_Performance':[gpu_tier],
        'Company':[brand],
        'CPU_Company':[cpu_brand],
        'CPU_Type':[cpu_type],
        'GPU_Company':[gpu_brand],
        'Resolution_Type':[resolution],
        'Storage_Type':[storage_type],
        'OpsSys_Grouped':[opsys]
    })

    onehot_cols = ["Company","CPU_Company","CPU_Type","GPU_Company",
                   "Resolution_Type","Storage_Type","OpsSys_Grouped"]
    data_input = pd.get_dummies(data_input, columns=onehot_cols, drop_first=True)

    try:
        model_columns = pickle.load(open('model_columns.pkl', 'rb'))
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model_columns.pkl: {e}")
        st.stop()

    data_input = data_input.reindex(columns=model_columns, fill_value=0)

    try:
        pred_log = model.predict(data_input)[0]
        pred_euro = np.expm1(pred_log)

        st.markdown("---")
        st.markdown(
            f"""
            <div style="background-color:#E8F0FE; padding:20px; border-radius:10px;">
                <h2 style="color:#1E90FF;">💰 Estimasi Harga Laptop</h2>
                <h1 style="color:#008000;">€ {pred_euro:,.2f}</h1>
                <p style="color:gray;">(Prediksi log-price: {pred_log:.4f})</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ✅ Simpan input user + hasil prediksi
        result = data_input.copy()
        result["Predicted_Price_Euro"] = pred_euro
        result["GPU_Performance_Level"] = gpu_level
        result["Brand"] = brand
        result["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_path = Path("user_predictions.csv")
        if file_path.exists():
            result.to_csv(file_path, mode="a", header=False, index=False)
        else:
            result.to_csv(file_path, index=False)

        st.success("✅ Data berhasil disimpan ke `user_predictions.csv`")

        # -------------------------
        # 📊 Tampilkan Chart Statistik
        # -------------------------
        st.markdown("## 📈 Statistik Prediksi Pengguna")
        df = pd.read_csv("user_predictions.csv")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.histogram(df, x="Predicted_Price_Euro", nbins=20, title="Distribusi Prediksi Harga (€)",
                                color_discrete_sequence=["#1E90FF"])
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            if "GPU_Performance_Level" in df.columns:
                avg_price = df.groupby("GPU_Performance_Level")["Predicted_Price_Euro"].mean().reset_index()
                fig2 = px.bar(avg_price, x="GPU_Performance_Level", y="Predicted_Price_Euro",
                              color="GPU_Performance_Level", title="Rata-rata Harga per GPU Tier",
                              color_discrete_sequence=["#32CD32", "#FFA500", "#FF4500"])
                st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📊 Lihat Data Prediksi yang Disimpan"):
            st.dataframe(df.tail(10))

    except Exception as e:
        st.exception(f"Error saat prediksi: {e}")
