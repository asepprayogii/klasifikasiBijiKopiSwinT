"""
Aplikasi Streamlit — Klasifikasi Jenis & Tingkat Sangrai Biji Kopi
Backbone : Swin Transformer Tiny (timm: swin_tiny_patch4_window7_224)
Pendekatan: Cascade
  Tahap 1 -> Jenis (Arabika, Ekselsa, Liberika, Robusta)
  Tahap 2 -> Tingkat Sangrai (Dark, Green, Light, Medium) memakai model spesifik per jenis

Jalankan dengan:  streamlit run app.py
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
MODEL_NAME = "swin_tiny_patch4_window7_224"
IMG_SIZE = 224

JENIS_LABELS = ["Arabika", "Ekselsa", "Liberika", "Robusta"]
ROAST_LABELS = ["Dark", "Green", "Light", "Medium"]

# Semua file bobot dipusatkan di folder weights/ — ganti nama file sesuai milik Anda.
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
JENIS_MODEL_PATH = os.path.join(WEIGHTS_DIR, "SwinT_Jenis_Fold4.pth")        # Val Acc 94.92%
ROAST_MODEL_PATHS = {
    "Arabika": os.path.join(WEIGHTS_DIR, "SwinT_Arabika_Fold2.pth"),         # Val Acc 100.00%
    "Ekselsa": os.path.join(WEIGHTS_DIR, "SwinT_Ekselsa_Fold2.pth"),         # Val Acc 100.00%
    "Liberika": os.path.join(WEIGHTS_DIR, "SwinT_Liberika_Fold2.pth"),       # Val Acc 100.00%
    "Robusta": os.path.join(WEIGHTS_DIR, "SwinT_Robusta_Fold1.pth"),         # Val Acc 100.00%
}

# Device otomatis: pakai GPU jika tersedia, jika tidak jatuh ke CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Arsitektur model (sama persis dengan notebook training)
# ----------------------------------------------------------------------------
class SwinCoffeeClassifier(nn.Module):
    """Wrapper backbone Swin-T (num_classes=0, sudah ter-pooling) + Custom Classifier Head.
    Dipakai untuk seluruh model: Jenis, Sangrai (per-jenis), maupun Flat — hanya num_classes yang berbeda.
    """

    def __init__(self, num_classes, pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=pretrained, num_classes=0, drop_rate=drop_rate
        )
        in_features = self.backbone.num_features  # 768 untuk Swin-T

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)       # (B, 768) — sudah ter-pooling
        return self.classifier(feats)  # (B, num_classes)


# ----------------------------------------------------------------------------
# Loader model (di-cache agar tidak di-load ulang tiap interaksi UI)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model(model_path, num_classes):
    """Muat satu checkpoint .pth ke SwinCoffeeClassifier dan kembalikan dalam mode eval."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")
    # pretrained=False: bobot ImageNet tidak perlu diunduh, karena akan ditimpa checkpoint.
    model = SwinCoffeeClassifier(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


# ----------------------------------------------------------------------------
# Preprocessing (identik dengan eval_tf pada notebook)
# ----------------------------------------------------------------------------
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_probs(model, tensor):
    """Kembalikan vektor probabilitas softmax (1D, numpy) dari satu model."""
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def make_bar_chart(labels, probs, title, color):
    df = pd.DataFrame({"Kelas": labels, "Probabilitas (%)": probs * 100})
    fig = px.bar(
        df, x="Kelas", y="Probabilitas (%)", text="Probabilitas (%)",
        title=title, color_discrete_sequence=[color],
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", marker_line_width=0)
    fig.update_yaxes(range=[0, 118], title=None, showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_xaxes(title=None)
    fig.update_layout(
        showlegend=False, height=330, margin=dict(t=55, b=10, l=10, r=10),
        title=dict(font=dict(size=15)), bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ----------------------------------------------------------------------------
# UI  (layout lebar / landscape: kiri = input, kanan = hasil)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Klasifikasi Biji Kopi", page_icon="☕", layout="wide")

# CSS: rapatkan margin atas/bawah & batasi tinggi pratinjau agar semua muat dalam satu layar landscape.
st.markdown(
    """
    <style>
      .block-container {padding-top: 3.2rem; padding-bottom: 0.8rem; max-width: 1500px;}
      h1 {font-size: 1.7rem !important; padding: 0 0 0.1rem 0 !important;}
      [data-testid="stImage"] img {max-height: 250px; width: auto; max-width: 100%;
                                   object-fit: contain; border-radius: 10px; margin: 0 auto; display: block;}
      [data-testid="stFileUploader"] section {padding: 0.6rem 0.8rem;}
      [data-testid="stMetric"] {background: rgba(128,128,128,0.08); border-radius: 10px; padding: 0.5rem 0.9rem;}
      [data-testid="stMetricValue"] {font-size: 1.5rem;}
      div[data-testid="stAlert"] h3 {margin: 0; padding: 0.1rem 0; font-size: 1.35rem; line-height: 1.5;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("☕ Klasifikasi Jenis & Tingkat Sangrai Biji Kopi")
st.caption(
    "Pendekatan *Cascade* berbasis Swin Transformer Tiny  •  Tahap 1: Jenis (Arabika, Ekselsa, Liberika, Robusta)  •  "
    f"Tahap 2: Tingkat Sangrai (Dark, Green, Light, Medium)  •  Perangkat: **{DEVICE.type.upper()}**"
)

col_in, col_out = st.columns([1, 2.4], gap="large")

# ------------------------------ Kolom kiri: input ------------------------------
with col_in:
    st.subheader("1. Unggah Citra", divider="gray")
    uploaded = st.file_uploader(
        "Unggah gambar biji kopi (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    image = None
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=f"{uploaded.name}  ({image.width}×{image.height} px)")
    predict_clicked = st.button("Prediksi", type="primary", use_container_width=True, disabled=image is None)

# ------------------------------ Kolom kanan: hasil -----------------------------
with col_out:
    st.subheader("2. Hasil Prediksi", divider="gray")

    if image is None:
        st.info("Silakan unggah gambar terlebih dahulu, lalu tekan **Prediksi**.")
    else:
        file_key = f"{uploaded.name}-{uploaded.size}"

        if predict_clicked:
            try:
                tensor = eval_tf(image).unsqueeze(0).to(DEVICE)

                # --- Tahap 1: Jenis ---
                model_jenis = load_model(JENIS_MODEL_PATH, num_classes=len(JENIS_LABELS))
                probs_jenis = predict_probs(model_jenis, tensor)
                pred_jenis = JENIS_LABELS[int(probs_jenis.argmax())]

                # --- Tahap 2: Tingkat Sangrai (model sesuai jenis hasil Tahap 1) ---
                model_roast = load_model(ROAST_MODEL_PATHS[pred_jenis], num_classes=len(ROAST_LABELS))
                probs_roast = predict_probs(model_roast, tensor)
                pred_roast = ROAST_LABELS[int(probs_roast.argmax())]

                st.session_state["hasil"] = {
                    "key": file_key, "jenis": pred_jenis, "roast": pred_roast,
                    "p_jenis": probs_jenis, "p_roast": probs_roast,
                }
            except FileNotFoundError as e:
                st.session_state.pop("hasil", None)
                st.error(f"{e}\n\nPastikan file .pth sudah diletakkan di folder `weights/` "
                         "dan namanya sesuai dengan konfigurasi di bagian atas `app.py`.")
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("hasil", None)
                st.error(f"Terjadi kesalahan saat inferensi: {e}")

        hasil = st.session_state.get("hasil")
        if hasil and hasil["key"] == file_key:
            st.success(f"### Hasil Prediksi: {hasil['jenis']} - {hasil['roast']} Roast")

            m1, m2 = st.columns(2)
            m1.metric("Jenis (Tahap 1)", hasil["jenis"], f"{hasil['p_jenis'].max() * 100:.2f}%")
            m2.metric("Tingkat Sangrai (Tahap 2)", hasil["roast"], f"{hasil['p_roast'].max() * 100:.2f}%")

            g1, g2 = st.columns(2)
            g1.plotly_chart(
                make_bar_chart(JENIS_LABELS, hasil["p_jenis"], "Probabilitas Jenis Kopi (Tahap 1)", "#6f4e37"),
                use_container_width=True,
            )
            g2.plotly_chart(
                make_bar_chart(ROAST_LABELS, hasil["p_roast"],
                               f"Probabilitas Tingkat Sangrai — model {hasil['jenis']} (Tahap 2)", "#c68642"),
                use_container_width=True,
            )
        else:
            st.info("Citra siap. Tekan tombol **Prediksi** untuk memulai klasifikasi.")