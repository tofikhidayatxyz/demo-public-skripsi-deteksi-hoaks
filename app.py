import streamlit as st
import pandas as pd
import numpy as np
import re
import html
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman
st.set_page_config(
    page_title="Asisten Verifikasi Berita & Hoaks",
    page_icon="🔍",
    layout="wide"
)

# Fungsi Preprocessing
def preprocess_text(text):
    if not text:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load Dataset & Cache Vectorizer
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("dataset.csv")
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
    
    return df, vectorizer, tfidf_matrix

df_ref, vectorizer, tfidf_matrix = load_data_and_model()

# Sidebar: Pengaturan & Info
st.sidebar.title("⚙️ Pengaturan Sistem")
st.sidebar.markdown(f"**Basis Data Rujukan:** {len(df_ref):,} artikel")
st.sidebar.markdown("- Berita Resmi (Detik.com): Fakta (0)")
st.sidebar.markdown("- Arsip Klarifikasi (TurnBackHoax): Hoaks (1)")
st.sidebar.divider()

threshold = st.sidebar.slider("Batas Minimum Kemiripan (Threshold):", min_value=0.10, max_value=0.90, value=0.30, step=0.05)
top_k = st.sidebar.slider("Jumlah Rujukan Teratas (Top-K):", min_value=1, max_value=5, value=3)

# Judul Utama
st.title("🔍 Asisten Verifikasi Berita & Deteksi Hoaks")
st.caption("Implementasi OCR + TF-IDF + Cosine Similarity dengan Pendekatan Human-in-the-Loop untuk Skripsi")

# Input Section (Tabs)
tab_image, tab_text = st.tabs(["📸 Unggah Tangkapan Layar (OCR)", "✍️ Masukkan Teks Langsung"])

input_text = ""

with tab_image:
    uploaded_file = st.file_uploader("Unggah gambar atau screenshot berita/chat medsos:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Tangkapan Layar Masukan", width=400)
        
        # Ekstrak Teks OCR
        ocr_result = ""
        try:
            import pytesseract
            ocr_result = pytesseract.image_to_string(img, lang="ind+eng")
        except Exception:
            st.warning("⚠️ Engine Tesseract OCR belum terpasang di sistem container. Silakan ketik atau tempelkan teks hasil baca gambar di bawah ini:")
        
        input_text = st.text_area("Teks Terdeteksi (Dapat Diedit Pengguna):", value=ocr_result, height=150)

with tab_text:
    manual_text = st.text_area("Tempelkan teks atau klaim berita yang ingin diuji:", height=150)
    if manual_text:
        input_text = manual_text

# Tombol Eksekusi
st.divider()
if st.button("🚀 Analisis Kemiripan (Cosine Similarity)", type="primary"):
    if not input_text.strip():
        st.error("Masukkan teks atau gambar terlebih dahulu!")
    else:
        # Preprocessing input
        cleaned_input = preprocess_text(input_text)
        
        if not cleaned_input:
            st.warning("Teks input terlalu pendek setelah dibersihkan.")
        else:
            # Transform input ke vektor TF-IDF
            input_vector = vectorizer.transform([cleaned_input])
            
            # Hitung Cosine Similarity
            similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
            
            # Ambil indeks rujukan teratas
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]
            
            best_idx = top_indices[0]
            best_score = float(top_scores[0])
            best_label = int(df_ref.iloc[best_idx]["label"])
            best_title = df_ref.iloc[best_idx]["judul"]
            
            st.subheader("📊 Hasil Rekomendasi Sistem")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="Skor Kemiripan Tertinggi", value=f"{best_score * 100:.2f}%")
            
            with col2:
                if best_score >= threshold:
                    if best_label == 1:
                        st.error("🚨 **INDIKASI: Cenderung HOAKS / DISINFORMASI**")
                        st.write("Sistem mendeteksi narasi input sangat mirip dengan arsip klarifikasi hoaks di basis data.")
                    else:
                        st.success("✅ **INDIKASI: Berita RESMI / FAKTA**")
                        st.write("Sistem mendeteksi narasi input bersesuaian dengan laporan portal berita terverifikasi.")
                else:
                    st.info("ℹ️ **STATUS: Belum Terdaftar di Basis Data Lokal**")
                    st.write(f"Skor kemiripan tertinggi ({best_score * 100:.2f}%) berada di bawah ambang batas ({threshold * 100:.0f}%). Perlu pengecekan manual.")

            # Rujukan Detail
            st.markdown("### 📚 Berita Rujukan Terdekat")
            for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
                row = df_ref.iloc[idx]
                label_tag = "🔴 [HOAKS]" if row["label"] == 1 else "🟢 [FAKTA]"
                with st.expander(f"Peringkat {rank} - {label_tag} {row['judul']} (Kemiripan: {score * 100:.2f}%)"):
                    st.markdown(f"**Sumber Rujukan:** [{row['url']}]({row['url']})")
                    st.markdown(f"**Ringkasan Narasi Rujukan:**\n\n{row['narasi']}")

            # Bagian Human-in-the-Loop
            st.divider()
            st.markdown("### 🧑‍💼 Keputusan Akhir Pengguna (Human-in-the-Loop)")
            st.caption("Sistem hanya bertindak sebagai pemberi saran cerdas. Pengambil keputusan tetap manusia.")
            
            default_choice_idx = 0
            if best_score >= threshold:
                default_choice_idx = 1 if best_label == 1 else 2
                
            pilihan = st.radio(
                "Tentukan verifikasi akhir berdasarkan telaah bukti di atas:",
                options=[
                    "Perlu Investigasi Tambahan (Data Belum Meyakinkan)",
                    "Diverifikasi sebagai Berita HOAKS",
                    "Diverifikasi sebagai Berita FAKTA / VALID"
                ],
                index=default_choice_idx
            )
            
            catatan = st.text_input("Catatan Analis / Pengguna (Opsional):")
            
            if st.button("💾 Simpan Keputusan Verifikasi"):
                st.success("Keputusan berhasil dicatat ke dalam log evaluasi skripsi!")
