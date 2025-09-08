import os
import io
import re
import string
import pandas as pd
import joblib
import pymysql
import pymysql.cursors
import geopandas as gpd
import folium
import nltk
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask import (
    Flask, request, send_file, render_template,
    redirect, url_for, flash, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from rapidfuzz import process, fuzz
from flask import Flask, render_template, request
from folium.plugins import MarkerCluster
import base64
from flask import request

# ================== Config ================== #
app = Flask(__name__)
app.secret_key = "supersecret"

ALLOWED_EXTS = {".csv", ".xls", ".xlsx"}

# ================== DB Connection ================== #
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="project",
    cursorclass=pymysql.cursors.DictCursor
)

# ================== Load Model & Geo ================== #
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
gdf = gpd.read_file("jakarta_kelurahan.geojson")

# Normalisasi nama kelurahan
gdf["nm_kelurahan"] = gdf["nm_kelurahan"].astype(str).str.upper().str.strip()
kelurahan_resmi = gdf["nm_kelurahan"].unique().tolist()

def preprocess_dataset(df):
    df["kelurahan"] = df["kelurahan"].astype(str).str.upper().str.strip()
    return df

# Mapping kategori (angka ke nama)
kategori_mapping = {
    1: "Pemerintahan",
    2: "Perekonomian dan Keuangan",
    3: "Pembangunan",
    4: "Kesejahteraan Rakyat",
    5: "Lainnya"
}

# Warna peta
warna_mapping = {
    "Pemerintahan": "red",
    "Perekonomian dan Keuangan": "blue",
    "Pembangunan": "green",
    "Kesejahteraan Rakyat": "purple",
    "Lainnya": "orange"
}

# Dataset global terakhir diproses (untuk download & peta)
df_global = None

# ================== NLP Preparation ================== #
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(nltk.corpus.stopwords.words("indonesian"))

def clean_text(text: str) -> str:
    """Cleaning text kolom Usulan."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = " ".join(w for w in text.split() if w not in stop_words)   # hapus stopwords
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalisasi_kelurahan(nama, daftar_resmi, threshold=78):
    """Fuzzy match nama kelurahan ke daftar resmi. Return tuple (hasil, flag)."""
    if pd.isna(nama) or str(nama).strip() == "":
        return None, "NA"
    nama_u = str(nama).upper().strip()
    hasil, skor, _ = process.extractOne(nama_u, daftar_resmi, scorer=fuzz.WRatio)
    return (hasil, "MATCH") if skor >= threshold else (None, f"LOW({skor})")

def preprocess_dataset(df: pd.DataFrame):
    """
    Pipeline preprocessing:
    - lower columns
    - drop kolom status (tapi TIDAK drop 'no' karena user mau tampilkan)
    - dropna
    - cleaning text kolom 'usulan'
    - prediksi kategori (angka + nama) -> hasil di kolom 'kategoriAngka' & 'kategori'
    - normalisasi kolom 'kelurahan' -> simpan ke kolom 'kelurahan' (overwritten dengan versi normal)
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    # hanya hapus 'status' jika ada; jangan hapus 'no'
    if "status" in df.columns:
        df.drop(columns="status", inplace=True)

    # pastikan kolom wajib ada
    if "usulan" not in df.columns:
        return None, "Kolom 'Usulan' tidak ditemukan pada dataset."

    # drop baris kosong
    df.dropna(how="all", inplace=True)
    df.dropna(subset=["usulan"], inplace=True)

    # cleaning text usulan
    df["usulan"] = df["usulan"].astype(str).apply(clean_text)

    # prediksi
    X = vectorizer.transform(df["usulan"].astype(str))
    y_pred = model.predict(X)

    df["kategoriAngka"] = y_pred
    df["kategori"] = df["kategoriAngka"].map(kategori_mapping)

    # simpan angka prediksi ke kolom 'kategoriAngka'
    if isinstance(y_pred[0], str):
        # model keluarkan teks -> buat mapping balik (teks -> angka)
        reverse_map = {v.lower(): k for k, v in kategori_mapping.items()}
        df["kategoriAngka"] = [reverse_map.get(str(x).lower(), 5) for x in y_pred]
    else:
        # model keluarkan angka
        df["kategoriAngka"] = [int(x) if pd.notna(x) else 5 for x in y_pred]

    # map ke nama kategori
    df["kategori"] = df["kategoriAngka"].map(kategori_mapping).fillna("Lainnya")

    # Normalisasi kelurahan: jika ada kolom 'kelurahan' dgn nilai user, normalisasi dan simpan kembali ke 'kelurahan'
    if "kelurahan" in df.columns:
        df["kelurahan"] = df["kelurahan"].astype(str).str.upper().str.strip()
        hasil = df["kelurahan"].apply(lambda x: normalisasi_kelurahan(x, kelurahan_resmi))
        # ambil hasil normal (atau None)
        df["kelurahan"] = hasil.apply(lambda t: t[0])
    else:
        df["kelurahan"] = None

    # Pastikan urutan kolom yang minimal & relevan (tidak menghapus kolom lain)
    minimal_order = []
    if "no" in df.columns:
        minimal_order.append("no")
    minimal_order.append("usulan")
    if "jenis" in df.columns:  # Tambah jenis ke urutan kolom
        minimal_order.append("jenis")
    minimal_order += ["kelurahan", "kategori", "kategoriAngka"]
    remaining = [c for c in df.columns if c not in minimal_order]
    df = df[minimal_order + remaining]

    return df, None
   
# ================== Routes ================== #
@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("admin_dashboard" if session.get("role") == "admin" else "predict"))
    return redirect(url_for("login"))

# -------- LOGIN -------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()

        if user and check_password_hash(user["password"], password):
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE users SET last_login = NOW() WHERE user_id = %s", 
                    (user["user_id"],)
                )
                conn.commit()

            session["user"] = user["username"]
            session["role"] = user.get("role", "user")
            flash("Login berhasil!", "success")
            return redirect(url_for("admin_dashboard" if session["role"] == "admin" else "predict"))

        flash("Email atau password salah", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")

# -------- FORGOT PASSWORD (Testing Mode) -------- #
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        
        if not email:
            flash('Email harus diisi!', 'danger')
            return redirect(url_for('forgot_password'))
        
        # Cek apakah email ada di database
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()
        
        if not user:
            flash('Email tidak terdaftar dalam sistem!', 'danger')
            return redirect(url_for('forgot_password'))
        
        # Generate token reset password
        reset_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=1)
        
        # Simpan token ke database
        try:
            with conn.cursor() as cursor:
                # Hapus token lama jika ada
                cursor.execute("DELETE FROM password_reset_tokens WHERE user_id=%s", (user['user_id'],))
                
                # Insert token baru
                cursor.execute(
                    "INSERT INTO password_reset_tokens (user_id, email, token, expires_at) VALUES (%s, %s, %s, %s)",
                    (user['user_id'], email, reset_token, expires_at)
                )
                conn.commit()
        except Exception as e:
            flash('Terjadi kesalahan sistem. Silakan coba lagi!', 'danger')
            print(f"Database error: {e}")
            return redirect(url_for('forgot_password'))
        
        # Generate reset URL
        reset_url = url_for('reset_password', token=reset_token, _external=True)
        
        # UNTUK TESTING: tampilkan link langsung
        print(f"🔗 RESET LINK: {reset_url}")
        print(f"📧 User: {user['username']} ({email})")
        print(f"⏰ Expires: {expires_at}")
        
        # Flash message dengan link (untuk testing)
        flash(f'Token reset berhasil dibuat! Link reset: <a href="{reset_url}" target="_blank" class="text-primary fw-bold">Reset Password</a>', 'success')
        
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    # Verifikasi token
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT prt.*, u.user_id, u.username 
            FROM password_reset_tokens prt
            JOIN users u ON prt.user_id = u.user_id
            WHERE prt.token=%s AND prt.expires_at > NOW()
        """, (token,))
        token_data = cursor.fetchone()
    
    if not token_data:
        flash('Link reset password tidak valid atau sudah kadaluarsa!', 'danger')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not password or not confirm_password:
            flash('Password dan konfirmasi password harus diisi!', 'danger')
            return render_template('reset_password.html', token=token)
        
        if password != confirm_password:
            flash('Password dan konfirmasi password tidak sama!', 'danger')
            return render_template('reset_password.html', token=token)
        
        if len(password) < 6:
            flash('Password minimal 6 karakter!', 'danger')
            return render_template('reset_password.html', token=token)
        
        # Update password user
        hashed_password = generate_password_hash(password)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE users SET password=%s, updated_at=NOW() WHERE user_id=%s",
                    (hashed_password, token_data['user_id'])
                )
                
                # Hapus token setelah berhasil reset
                cursor.execute("DELETE FROM password_reset_tokens WHERE token=%s", (token,))
                conn.commit()
            
            flash('Password berhasil direset! Silakan login dengan password baru.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            flash('Terjadi kesalahan sistem. Silakan coba lagi!', 'danger')
            return render_template('reset_password.html', token=token)
    
    return render_template('reset_password.html', token=token)

# -------- REGISTER -------- #
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confpassword = request.form.get("confpassword")

        if not username or not email or not password:
            flash("Lengkapi semua field.", "danger")
            return redirect(url_for("register"))

        if password != confpassword:
            flash("Password tidak sama!", "danger")
            return redirect(url_for("register"))

        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            if cursor.fetchone():
                flash("Email sudah terdaftar!", "danger")
                return redirect(url_for("register"))

            cursor.execute(
                "INSERT INTO users (username, email, password, role) VALUES (%s, %s, %s, %s)",
                (username, email, generate_password_hash(password), "user")
            )
            conn.commit()

        flash("Registrasi berhasil, silakan login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# -------- PREDICT / UPLOAD -------- #
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        flash("Silakan login terlebih dahulu", "danger")
        return redirect(url_for("login"))
     
    global df_global
    if 'last_processed_user' not in session or session['last_processed_user'] != session['user']:
        df_global = None
        session['last_processed_user'] = session['user']
    
    selected_kategori = request.args.get("filter_kategori", "")
    selected_jenis = request.args.get("filter_jenis", "")

    if request.method == "POST":
        file = request.files.get("dataset")
        tahun = request.form.get("tahun")

        if not file or not tahun:
            flash("Tidak ada file yang diupload atau tahun belum dipilih", "danger")
            return redirect(url_for("predict"))

        filename = secure_filename(file.filename)

        # Validasi format file
        if not filename.lower().endswith(('.csv', '.xls', '.xlsx')):
            flash("Format file tidak didukung. Gunakan CSV, XLS, atau XLSX", "danger")
            return redirect(url_for("predict"))

        # Ambil user_id dari session
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_id FROM users WHERE username=%s", (session["user"],))
            user_data = cursor.fetchone()
            if not user_data:
                flash("User tidak ditemukan", "danger")
                return redirect(url_for("login"))
            user_id = user_data["user_id"]

        # Load file langsung dari memory
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
        except Exception as e:
            flash(f"Gagal membaca file: {e}", "danger")
            return redirect(url_for("predict"))

        # Preprocess dataset
        df, error = preprocess_dataset(df)
        if error:
            flash(error, "danger")
            return redirect(url_for("predict"))
        
        # Penyimpanan Data
        file_type = os.path.splitext(filename)[1].lower().replace(".", "")
        uploaded_at = datetime.now()

        try:
            with conn.cursor() as cursor:
                # 1. INSERT KE TABEL UPLOADS
                print(f"DEBUG: Inserting upload record for user_id={user_id}")
                cursor.execute(
                    "INSERT INTO uploads (user_id, nama_file, file_type, tahun, uploaded_at) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, filename, file_type, tahun, uploaded_at)
                )
                upload_id = cursor.lastrowid  # Ambil ID yang baru saja di-insert
                print(f"DEBUG: Upload record created with ID={upload_id}")
                
                # 2. VERIFIKASI UPLOAD RECORD TERSIMPAN
                cursor.execute("SELECT * FROM uploads WHERE id = %s", (upload_id,))
                upload_record = cursor.fetchone()
                print(f"DEBUG: Upload record verification: {upload_record}")
                
                if not upload_record:
                    raise Exception("Upload record tidak tersimpan dengan benar")

                # 3. INSERT BATCH KE TABEL USULAN
                rows = []
                for idx, r in df.iterrows():
                    # Pastikan semua nilai tidak None dan tipe data benar
                    usulan_text = str(r["usulan"]) if pd.notna(r["usulan"]) else ""
                    kelurahan = str(r["kelurahan"]) if pd.notna(r["kelurahan"]) and r["kelurahan"] not in [None, "None", ""] else None
                    jenis = str(r["jenis"]) if "jenis" in r and pd.notna(r["jenis"]) and r["jenis"] not in [None, "None", ""] else None
                    kategori = str(r["kategori"]) if pd.notna(r["kategori"]) else "Lainnya"
                    kategori_angka = int(r["kategoriAngka"]) if pd.notna(r["kategoriAngka"]) else 5
                    
                    row_data = (upload_id, usulan_text, kelurahan, jenis, kategori, kategori_angka)
                    rows.append(row_data)
                    
                print(f"DEBUG: Prepared {len(rows)} rows for insertion")
                print(f"DEBUG: Sample row: {rows[0] if rows else 'No rows'}")
                
                # Insert dengan batch
                cursor.executemany(
                    "INSERT INTO usulan (upload_id, usulan, kelurahan, jenis, kategori, kategoriAngka) VALUES (%s, %s, %s, %s, %s, %s)",
                    rows
                )
                
                # 4. COMMIT SEMUA PERUBAHAN
                conn.commit()
                print("DEBUG: All data committed successfully")
                
                # 5. VERIFIKASI DATA TERSIMPAN
                cursor.execute("SELECT COUNT(*) as count FROM usulan WHERE upload_id = %s", (upload_id,))
                result = cursor.fetchone()
                usulan_count = result['count'] if result else 0
                print(f"DEBUG: Usulan records saved: {usulan_count}")
                
                cursor.execute("SELECT COUNT(*) as total FROM usulan")
                total_result = cursor.fetchone()
                total_count = total_result['total'] if total_result else 0
                print(f"DEBUG: Total usulan records in database: {total_count}")
                
                if usulan_count == 0:
                    raise Exception("Tidak ada data usulan yang tersimpan")

            # Set global df dan flash success
            df_global = df
            session['last_processed_user'] = session['user']
            flash(f"File berhasil diupload dan diproses! {len(df)} records tersimpan.", "success")
            return render_template("predict.html", results=df.to_dict(orient="records"))

        except Exception as e:
            # Rollback jika ada error
            conn.rollback()
            flash(f"Gagal menyimpan hasil prediksi: {e}", "danger")
            print(f"DEBUG Error: {e}")
            import traceback
            traceback.print_exc()
            return redirect(url_for("predict"))

    # Jika GET, tampilkan hasil prediksi terakhir (jika ada)
    if df_global is not None:
        df_filtered = df_global

        # Terapkan filter kategori
        if selected_kategori:
            df_filtered = df_global[df_global["kategori"] == selected_kategori]

        # Terapkan filter jenis
        if selected_jenis and "jenis" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["jenis"] == selected_jenis]

        return render_template("predict.html", 
                             results=df_filtered.to_dict(orient="records"),
                             selected_kategori=selected_kategori, 
                             selected_jenis=selected_jenis)
    else:
        return render_template("predict.html", 
                             results=None, 
                             selected_kategori=selected_kategori, 
                             selected_jenis=selected_jenis)
# -------- LOGOUT -------- #
@app.route("/logout")
def logout():
    global df_global
    df_global = None
    session.clear()
    flash("Logout berhasil", "success")
    return redirect(url_for("login"))

# -------- ADMIN DASHBOARD -------- #
@app.route("/admin_dashboard")
def admin_dashboard():
    if "user" not in session or session.get("role") != "admin":
        flash("Akses ditolak. Hanya admin yang dapat mengakses halaman ini.", "danger")
        return redirect(url_for("login"))

    print("DEBUG DASHBOARD: Starting admin dashboard")
    
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # Debug: Cek data di setiap tabel
            cursor.execute("SELECT COUNT(*) as total FROM users")
            user_count = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(*) as total FROM uploads")
            upload_count = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(*) as total FROM usulan")
            usulan_count = cursor.fetchone()['total']
            
            print(f"DEBUG DASHBOARD: users={user_count}, uploads={upload_count}, usulan={usulan_count}")
            
            # Main query dengan penanganan error yang lebih baik
            query = """
            SELECT 
                us.id as usulan_id,
                us.usulan,
                us.kelurahan,
                us.jenis,
                us.kategori,
                us.kategoriAngka,
                up.nama_file,
                up.tahun,
                up.uploaded_at,
                u.username,
                u.user_id
            FROM usulan us 
            JOIN uploads up ON us.upload_id = up.id 
            JOIN users u ON up.user_id = u.user_id
            ORDER BY up.uploaded_at DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Query untuk mendapatkan tahun terakhir
            cursor.execute("""
                SELECT tahun 
                FROM uploads 
                ORDER BY uploaded_at DESC 
                LIMIT 1
            """)
            tahun_terakhir_result = cursor.fetchone()
            tahun_terakhir = tahun_terakhir_result['tahun'] if tahun_terakhir_result else "Belum ada data"
            
            print(f"DEBUG DASHBOARD: Tahun terakhir: {tahun_terakhir}")
            
        print(f"DEBUG DASHBOARD: Retrieved {len(results)} records from database")

        # Convert to DataFrame for analysis
        if results:
            df = pd.DataFrame(results)
            print(f"DEBUG DASHBOARD: DataFrame shape: {df.shape}")
            print(f"DEBUG DASHBOARD: Columns: {df.columns.tolist()}")
            
            # Print sample data
            if len(df) > 0:
                print(f"DEBUG DASHBOARD: Sample record:")
                print(f"  - usulan: {df.iloc[0]['usulan'][:50]}...")
                print(f"  - kelurahan: {df.iloc[0]['kelurahan']}")
                print(f"  - kategori: {df.iloc[0]['kategori']}")
                print(f"  - tahun: {df.iloc[0]['tahun']}")
                
        else:
            df = pd.DataFrame()
            print("DEBUG DASHBOARD: No records found, creating empty DataFrame")

        # Calculate statistics
        if df.empty:
            print("DEBUG DASHBOARD: DataFrame is empty, using default values")
            total_usulan = 0
            jumlah_kelurahan = 0
            kategori_terbanyak = None
            distribusi_kategori = {}
            distribusi_kelurahan = {}
            distribusi_tahun = {}
        else:
            total_usulan = len(df)
            print(f"DEBUG DASHBOARD: Total usulan: {total_usulan}")
            
            # Kelurahan analysis
            kelurahan_valid = df[
                df["kelurahan"].notna() & 
                (df["kelurahan"] != "") & 
                (df["kelurahan"] != "None") & 
                (df["kelurahan"] != "TIDAK DIKETAHUI")
            ]
            jumlah_kelurahan = kelurahan_valid["kelurahan"].nunique() if not kelurahan_valid.empty else 0
            print(f"DEBUG DASHBOARD: Valid kelurahan count: {jumlah_kelurahan}")
            
            # Kategori analysis
            if "kategori" in df.columns and not df["kategori"].empty:
                kategori_counts = df["kategori"].value_counts()
                kategori_terbanyak = kategori_counts.index[0] if not kategori_counts.empty else None
                distribusi_kategori = kategori_counts.to_dict()
                print(f"DEBUG DASHBOARD: Kategori terbanyak: {kategori_terbanyak}")
                print(f"DEBUG DASHBOARD: Distribusi kategori: {distribusi_kategori}")
            else:
                kategori_terbanyak = None
                distribusi_kategori = {}

            # Distribusi kelurahan (top 10)
            if not kelurahan_valid.empty:
                distribusi_kelurahan = kelurahan_valid["kelurahan"].value_counts().head(10).to_dict()
                print(f"DEBUG DASHBOARD: Top kelurahan: {list(distribusi_kelurahan.keys())[:3]}")
            else:
                distribusi_kelurahan = {}

            # Tahun analysis untuk chart
            if "tahun" in df.columns:
                df["tahun_clean"] = pd.to_numeric(df["tahun"], errors='coerce')
                tahun_valid = df[df["tahun_clean"].notna()]
                if not tahun_valid.empty:
                    distribusi_tahun = tahun_valid["tahun_clean"].value_counts().sort_index().to_dict()
                    distribusi_tahun = {int(k): v for k, v in distribusi_tahun.items()}
                    print(f"DEBUG DASHBOARD: Distribusi tahun: {distribusi_tahun}")
                else:
                    distribusi_tahun = {}
            else:
                distribusi_tahun = {}

        print("DEBUG DASHBOARD: Rendering template with calculated statistics")

        return render_template(
            "admin_dashboard.html",
            total_usulan=total_usulan,
            total_kelurahan=jumlah_kelurahan,
            kategori_terbanyak=kategori_terbanyak,
            tahun_terakhir=tahun_terakhir,
            kategori_labels=list(distribusi_kategori.keys()),
            kategori_counts=list(distribusi_kategori.values()),
            kelurahan_labels=list(distribusi_kelurahan.keys()),
            kelurahan_counts=list(distribusi_kelurahan.values()),
            tahun_labels=list(distribusi_tahun.keys()),
            tahun_counts=list(distribusi_tahun.values()),
        )
        
    except Exception as e:
        print(f"DEBUG DASHBOARD: Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        flash(f"Terjadi kesalahan saat mengambil data dashboard: {str(e)}", "danger")
        
        # Return template with empty data if error
        return render_template(
            "admin_dashboard.html",
            total_usulan=0,
            total_kelurahan=0,
            kategori_terbanyak=None,
            tahun_terakhir="Belum ada data",  # ===== TAMBAHAN: Default value =====
            kategori_labels=[],
            kategori_counts=[],
            kelurahan_labels=[],
            kelurahan_counts=[],
            tahun_labels=[],
            tahun_counts=[],
        )

@app.route("/debug/check_data")
def debug_check_data():
    """Route khusus untuk debugging data"""
    if "user" not in session or session.get("role") != "admin":
        return "Access denied"
    
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # Check all tables
            tables_info = {}
            
            for table in ['users', 'uploads', 'usulan']:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                count = cursor.fetchone()['count']
                
                cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                sample = cursor.fetchall()
                
                tables_info[table] = {
                    'count': count,
                    'sample': sample
                }
            
            # Check foreign key relationships
            cursor.execute("""
                SELECT us.*, up.nama_file, u.username 
                FROM usulan us 
                LEFT JOIN uploads up ON us.upload_id = up.id 
                LEFT JOIN users u ON up.user_id = u.user_id 
                LIMIT 10
            """)
            join_test = cursor.fetchall()
            
            return f"""
            <h2>Database Debug Info</h2>
            <h3>Tables Info:</h3>
            <pre>{tables_info}</pre>
            
            <h3>Join Test Result:</h3>
            <pre>{join_test}</pre>
            """
            
    except Exception as e:
        return f"Error: {e}"


@app.route("/admin/users")
def manage_users():
    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute("""
            SELECT user_id, username, email, role, created_at, updated_at, last_login 
            FROM users
        """)
        users = cursor.fetchall()
    return render_template("manage_users.html", users=users)

@app.route("/admin/users/edit/<int:user_id>", methods=["GET", "POST"])
def edit_user(user_id):
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        role = request.form["role"]

        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE users
                SET username=%s, email=%s, role=%s, updated_at=NOW()
                WHERE user_id=%s
            """, (username, email, role, user_id))
            conn.commit()
        flash("User berhasil diupdate", "success")
        return redirect(url_for("manage_users"))

    # kalau GET → ambil data user buat form edit
    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
        user = cursor.fetchone()

    return render_template("edit_user.html", user=user)


@app.route("/admin/users/delete/<int:user_id>", methods=["POST", "GET"])
def delete_user(user_id):
    with conn.cursor() as cursor:
        # Hapus semua data terkait dulu sesuai urutan dependency
        cursor.execute("DELETE FROM usulan WHERE upload_id IN (SELECT id FROM uploads WHERE user_id=%s)", (user_id,))
        cursor.execute("DELETE FROM uploads WHERE user_id=%s", (user_id,))
        cursor.execute("DELETE FROM password_reset_tokens WHERE user_id=%s", (user_id,))
        cursor.execute("DELETE FROM users WHERE user_id=%s", (user_id,))
        conn.commit()
    flash("User berhasil dihapus", "success")
    return redirect(url_for("manage_users"))

@app.route("/admin/uploads")
def manage_uploads():
    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
        # JOIN dengan tabel users untuk menampilkan nama user
        cursor.execute("""
            SELECT u.*, us.username 
            FROM uploads u 
            LEFT JOIN users us ON u.user_id = us.user_id 
            ORDER BY u.uploaded_at DESC
        """)
        uploads = cursor.fetchall()
    return render_template("manage_uploads.html", uploads=uploads)


@app.route("/admin/uploads/delete/<int:upload_id>", methods=["POST", "GET"])
def delete_upload(upload_id):
    # Hapus juga file fisiknya
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM uploads WHERE id=%s", (upload_id,))
        conn.commit()
    flash("Upload berhasil dihapus", "success")
    return redirect(url_for("manage_uploads"))

# admin membuat user
@app.route("/create_user", methods=["GET", "POST"])
def create_user():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]

        # Cek apakah email sudah ada
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            if cursor.fetchone():
                flash("Email sudah terdaftar!", "danger")
                return redirect(url_for("create_user"))

            # Hash password dan simpan ke database
            hashed_password = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, email, password, role) VALUES (%s, %s, %s, %s)",
                (username, email, hashed_password, role)
            )
            conn.commit()

        flash("Pengguna berhasil ditambahkan!", "success")
        return redirect(url_for("manage_users"))

    return render_template("create_user.html")

# -------- DOWNLOAD HASIL -------- #
@app.route("/download")
def download_file():
    global df_global
    if df_global is None:
        flash("Belum ada dataset diproses", "danger")
        return redirect(url_for("predict"))

    output = io.BytesIO()
    df_global.to_excel(output, index=False)
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="hasil_klasifikasi.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -------- MAP (Folium) -------- #
@app.route("/map")
def show_map():
    global df_global
    if df_global is None:
        flash("Belum ada dataset yang diproses", "danger")
        return redirect(url_for("predict"))

    selected_kategori = request.args.get("filter_kategori", "")
    selected_jenis = request.args.get("filter_jenis", "")

    df = df_global.copy()

    if not {"kelurahan", "kategori"}.issubset(df.columns):
        return "Dataset harus punya kolom: kelurahan, kategori"

    # Terapkan filter kategori
    if selected_kategori:
        df = df[df["kategori"] == selected_kategori]

    # Terapkan filter jenis
    if selected_jenis and "jenis" in df.columns:
        df = df[df["jenis"] == selected_jenis]

    # Hitung jumlah usulan per kelurahan & kategori
    counts = (
        df.groupby(["kelurahan", "kategori"])
        .size()
        .reset_index(name="jumlah")
    )

    # Ambil kategori utama (yang jumlahnya paling banyak)
    utama = (
        counts.sort_values(["kelurahan", "jumlah"], ascending=[True, False])
        .drop_duplicates("kelurahan")
        .rename(columns={"kategori": "kategori_utama"})
    )
    merged = gdf.merge(
        utama[["kelurahan", "kategori_utama"]],
        left_on="nm_kelurahan", right_on="kelurahan",
        how="left"
    )

    # Buat dict kelurahan -> detail kategori (popup)
    popup_dict = {}
    for kel, group in df.groupby("kelurahan"):
        kategori_counts = group["kategori"].value_counts()

        if "jenis" in group.columns:
            jenis_counts = group["jenis"].value_counts()

        # Urutkan dari yang terbesar ke terkecil
        kategori_counts = kategori_counts.sort_values(ascending=False)

        detail = "".join(
            [f"{kat}: {jum} usulan<br>" for kat, jum in kategori_counts.items()]
        )

        if "jenis" in group.columns and not jenis_counts.empty:
            detail += "<hr><b>Berdasarkan Jenis:</b><br>"
            detail += "".join(
                [f"{jenis}: {jum} usulan<br>" for jenis, jum in jenis_counts.items()]
            )

        popup_html = f"<b>{kel}</b><br>{detail}"
        popup_dict[kel] = popup_html

    # Peta
    m = folium.Map(location=[-6.2, 106.8], zoom_start=11, tiles="CartoDB positron")

    for _, row in merged.iterrows():
        kel = row["nm_kelurahan"]
        kategori = row.get("kategori_utama", None)
        warna = warna_mapping.get(kategori, "gray")

        popup_html = popup_dict.get(kel, f"<b>{kel}</b><br>Tidak ada usulan")

        folium.GeoJson(
            row["geometry"],
            style_function=lambda x, warna=warna: {
                "fillColor": warna,
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.6,
            },
            # Tooltip pakai detail juga
            tooltip=folium.Tooltip(popup_dict.get(kel, kel), sticky=True),
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)

    map_html = m._repr_html_()
    return render_template(
        "map.html", 
        map_html=map_html,
        selected_kategori=selected_kategori,
        selected_jenis=selected_jenis
    )

if __name__ == "__main__":
    app.run(debug=True)