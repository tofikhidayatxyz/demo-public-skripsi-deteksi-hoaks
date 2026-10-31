# 🔍 Sistem Asisten Verifikasi Berita & Deteksi Hoaks (Skripsi)

Sistem bantuan keputusan untuk mendeteksi potensi hoaks/disinformasi pada tangkapan layar atau teks berita menggunakan metode **OCR + TF-IDF + Cosine Similarity** dengan paradigma **Human-in-the-Loop**.

---

## 🚀 Cara Menjalankan di Replit

1. Buka link Repl atau import repositori ini langsung dari Replit.
2. Replit akan otomatis membaca file `.replit` dan memasang dependensi dari `requirements.txt`.
3. Klik tombol **Run** di bagian atas.
4. Aplikasi Streamlit akan otomatis muncul di panel *Webview* Replit.

---

## 💻 Cara Menjalankan di Komputer Lokal

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/tofikhidayatxyz/skripsi-deteksi-hoaks.git
   cd skripsi-deteksi-hoaks
   ```

2. **Buat dan aktifkan virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```

5. Buka browser di alamat `http://localhost:8501`.

---

## 📂 Struktur Berkas

- `app.py`: Antarmuka web interaktif (Streamlit) & logika pencocokan Cosine Similarity.
- `dataset.csv`: Basis data rujukan berita resmi (Detik.com, Label: 0) dan klarifikasi hoaks (TurnBackHoax, Label: 1).
- `requirements.txt`: Daftar pustaka Python yang dibutuhkan.
- `.replit` & `replit.nix`: Konfigurasi otomatis untuk runtime Replit.
