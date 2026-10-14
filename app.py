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
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_yaxes(range=[0, 110])
    fig.update_layout(showlegend=False, margin=dict(t=60, b=20))
    return fig


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Klasifikasi Biji Kopi", page_icon="☕", layout="centered")

st.title("☕ Klasifikasi Jenis & Tingkat Sangrai Biji Kopi")
st.write(
    "Unggah foto biji kopi untuk memprediksi **jenis** (Arabika, Ekselsa, Liberika, Robusta) "
    "dan **tingkat sangrai** (Dark, Green, Light, Medium) menggunakan pendekatan *Cascade* "
    "berbasis Swin Transformer Tiny."
)
st.caption(f"Perangkat inferensi: **{DEVICE.type.upper()}**")

uploaded = st.file_uploader("Unggah gambar biji kopi (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    if st.button("Prediksi", type="primary"):
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

            # --- Hasil ---
            st.success(f"### Hasil Prediksi: {pred_jenis} - {pred_roast} Roast")
            c1, c2 = st.columns(2)
            c1.metric("Jenis", pred_jenis, f"{probs_jenis.max() * 100:.2f}%")
            c2.metric("Tingkat Sangrai", pred_roast, f"{probs_roast.max() * 100:.2f}%")

            st.plotly_chart(
                make_bar_chart(JENIS_LABELS, probs_jenis, "Probabilitas Jenis Kopi (Tahap 1)", "#6f4e37"),
                use_container_width=True,
            )
            st.plotly_chart(
                make_bar_chart(ROAST_LABELS, probs_roast,
                               f"Probabilitas Tingkat Sangrai — model {pred_jenis} (Tahap 2)", "#c68642"),
                use_container_width=True,
            )
        except FileNotFoundError as e:
            st.error(f"{e}\n\nPastikan file .pth sudah diletakkan di folder `weights/` "
                     "dan namanya sesuai dengan konfigurasi di bagian atas `app.py`.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Terjadi kesalahan saat inferensi: {e}")
else:
    st.info("Silakan unggah gambar terlebih dahulu.")
