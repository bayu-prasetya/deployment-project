import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from model.predict import predict
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from model.bundle import load_bundle



# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Fraud Detector",
    page_icon="🛡️",
    layout="wide",
)

# ── Load bundle ─────────────────────────────────────────────────────────────
@st.cache_resource
def load():
    try:
        bundle       = load_bundle('model/artifacts/bundle_v1.sav')
        pipeline     = bundle["pipeline"]
        preprocessor = pipeline[:-1]
        model        = pipeline[-1]
        X_train_tr   = bundle["X_train_transformed"]
        features_raw = bundle["features"]
        features_tr  = bundle["feature_names_transformed"]
        explainer    = shap.LinearExplainer(model, X_train_tr)
        return pipeline, preprocessor, model, explainer, features_raw, features_tr
    except FileNotFoundError:
        st.error("Bundle model belum ditemukan.")
        st.info("Jalankan training terlebih dahulu:\n\n```bash\npython -m model.train\n```")
        st.stop()

pipeline, preprocessor, model, explainer, features_raw, features_tr = load()

# ── Helpers ────────────────────────────────────────────────────────────────

def risk_level(prob: float):
    if prob < 0.25:
        return "Rendah",       "🟢", "Klaim tampak legitimate. Proses klaim sesuai SOP standar.",
    elif prob < 0.50:
        return "Sedang",       "🟡", "Terdapat beberapa indikasi mencurigakan. Lakukan verifikasi dokumen tambahan sebelum menyetujui klaim."
    elif prob < 0.75:
        return "Tinggi",       "🟠", "Risiko fraud signifikan. Eskalasi ke tim investigasi dan tahan pembayaran sementara investigasi berlangsung."
    else:
        return "Sangat Tinggi","🔴", "Indikasi fraud kuat. Tolak sementara klaim, laporkan ke unit SIU (Special Investigation Unit), dan dokumentasikan seluruh bukti."

def get_shap(raw_df: pd.DataFrame) -> np.ndarray:
    transformed = preprocessor.transform(raw_df)
    return explainer.shap_values(transformed)

def shap_bar_chart(raw_df: pd.DataFrame, title: str = ""):
    sv    = get_shap(raw_df)
    vals  = sv[0]
    names = features_tr   # nama fitur setelah transform

    sorted_idx = np.argsort(np.abs(vals))[-12:]
    v = vals[sorted_idx]
    n = [names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d62728" if x > 0 else "#1f77b4" for x in v]
    ax.barh(n, v, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value  (+ = meningkatkan risiko fraud)")
    ax.set_title(title or "Kontribusi Fitur terhadap Prediksi")
    plt.tight_layout()
    return fig

# ── Session state ──────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "batch_df" not in st.session_state:
    st.session_state.batch_df = None
if "batch_result" not in st.session_state:
    st.session_state.batch_result = None
if "detail_idx" not in st.session_state:
    st.session_state.detail_idx = None

# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Fraud Detector")
    st.markdown("---")
    if st.button("🏠  Beranda",          use_container_width=True):
        st.session_state.page = "home"
    if st.button("🔍  Single Prediction", use_container_width=True):
        st.session_state.page = "single"
    if st.button("📂  Batch Prediction",  use_container_width=True):
        st.session_state.page = "batch"
        st.session_state.detail_idx = None
    st.markdown("---")
    st.caption("Insurance Fraud Detector v1.0")

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════
if page == "home":
    st.title("🛡️ Insurance Fraud Detection System")
    st.markdown(
        "Sistem deteksi fraud asuransi berbasis machine learning untuk membantu "
        "analis klaim mengidentifikasi potensi penipuan secara cepat dan akurat."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Single Prediction")
        st.write(
            "Masukkan data klaim secara manual dan dapatkan prediksi probabilitas fraud "
            "beserta penjelasan kontribusi setiap fitur (Explainable AI)."
        )
        if st.button("Mulai Single Prediction →", use_container_width=True):
            st.session_state.page = "single"
            st.rerun()

    with col2:
        st.subheader("📂 Batch Prediction")
        st.write(
            "Upload file CSV berisi banyak klaim sekaligus. Hasil prediksi ditampilkan "
            "dalam tabel interaktif — klik baris untuk melihat detail dan penjelasan AI."
        )
        if st.button("Mulai Batch Prediction →", use_container_width=True):
            st.session_state.page = "batch"
            st.rerun()

    st.markdown("---")
    st.subheader("📚 Apa itu Insurance Fraud?")

    st.markdown(
        """
        **Insurance fraud** adalah tindakan disengaja untuk mendapatkan pembayaran klaim
        asuransi secara tidak sah. Fraud asuransi merugikan industri secara masif —
        diperkirakan menyebabkan kerugian miliaran dolar per tahun secara global.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "**🚗 Staged Accidents**\n\n"
            "Kecelakaan yang disengaja atau dibuat-buat untuk mengajukan klaim asuransi kendaraan."
        )
    with c2:
        st.info(
            "**📋 Inflated Claims**\n\n"
            "Melaporkan nilai kerugian yang lebih besar dari kenyataan, misalnya harga kendaraan yang dimanipulasi."
        )
    with c3:
        st.info(
            "**👤 Identity Fraud**\n\n"
            "Menggunakan identitas orang lain atau membuat polis fiktif untuk mendapatkan pembayaran klaim."
        )

    st.markdown("---")
    st.subheader("🚩 Red Flags Umum dalam Klaim")

    col_a, col_b = st.columns(2)
    with col_a:
        st.warning(
            "- Klaim diajukan sangat cepat setelah polis dibuat\n"
            "- Tidak ada saksi dalam kejadian kecelakaan\n"
            "- Riwayat klaim yang sangat banyak sebelumnya\n"
            "- Kecelakaan terjadi di lokasi terpencil (Highway/Parking Lot)"
        )
    with col_b:
        st.warning(
            "- Perubahan alamat mendekati tanggal klaim\n"
            "- Nilai estimasi klaim tidak proporsional dengan harga kendaraan\n"
            "- Laporan polisi tidak diajukan meski klaim besar\n"
            "- Persentase liability yang tidak wajar"
        )

    st.markdown("---")
    st.subheader("⚙️ Tentang Model")
    st.markdown(
        """
        Model menggunakan **Random Forest Classifier** yang dilatih pada data historis klaim asuransi.
        Explainability menggunakan **SHAP (SHapley Additive exPlanations)** untuk menjelaskan
        kontribusi setiap fitur terhadap prediksi secara transparan.

        | Metrik | Nilai |
        |--------|-------|
        | Algoritma | Random Forest (100 trees) |
        | Akurasi (test set) | 83.4% |
        | Fitur yang digunakan | 21 fitur |
        | Explainability | SHAP Tree Explainer |
        """
    )

# ══════════════════════════════════════════════════════════════════════════
# PAGE: SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "single":
    st.title("🔍 Single Prediction")
    st.markdown("Isi form di bawah ini untuk memprediksi potensi fraud pada satu klaim.")
    st.markdown("---")
 
    with st.form("single_form"):
        st.subheader("👤 Profil Pengemudi")
        c1, c2, c3 = st.columns(3)
        age_of_driver    = c1.number_input("Usia Pengemudi", 16, 100, 35)
        gender           = c2.selectbox("Gender", ["M", "F"])
        marital_status   = c3.selectbox("Status Pernikahan", [1.0, 0.0], format_func=lambda x: "Menikah" if x == 1.0 else "Belum Menikah")
 
        c1, c2, c3 = st.columns(3)
        safty_rating      = c1.slider("Safety Rating", 0, 100, 75)
        annual_income     = c2.number_input("Pendapatan Tahunan (USD)", 0, 500000, 60000, step=1000)
        living_status     = c3.selectbox("Status Tempat Tinggal", ["Own", "Rent"])
 
        c1, c2, c3 = st.columns(3)
        high_education_ind = c1.selectbox("Pendidikan Tinggi", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        address_change_ind = c2.selectbox("Perubahan Alamat", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        zip_code           = c3.number_input("Kode Pos", 0, 99999, 85027)
 
        st.subheader("📋 Detail Klaim")
        c1, c2, c3 = st.columns(3)
        claim_date         = c1.text_input("Tanggal Klaim (M/D/YYYY)", value="1/1/2024")
        claim_day_of_week  = c2.selectbox("Hari Klaim", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        accident_site      = c3.selectbox("Lokasi Kecelakaan", ["Local", "Parking Lot", "Highway"])
 
        c1, c2, c3 = st.columns(3)
        past_num_of_claims    = c1.number_input("Jumlah Klaim Sebelumnya", 0, 50, 0)
        witness_present_ind   = c2.selectbox("Saksi Hadir", [1.0, 0.0], format_func=lambda x: "Ya" if x == 1.0 else "Tidak")
        liab_prct             = c3.slider("Liability (%)", 0, 100, 50)
 
        c1, c2 = st.columns(2)
        policy_report_filed_ind = c1.selectbox("Laporan Polisi Diajukan", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        claim_est_payout        = c2.number_input("Estimasi Payout Klaim (USD)", 0.0, 100000.0, 5000.0, step=100.0)
 
        st.subheader("🚗 Informasi Kendaraan")
        c1, c2, c3 = st.columns(3)
        age_of_vehicle   = c1.number_input("Usia Kendaraan (tahun)", 0, 30, 5)
        vehicle_category = c2.selectbox("Kategori Kendaraan", ["Compact", "Medium", "Large"])
        vehicle_color    = c3.selectbox("Warna Kendaraan", ["other","blue","black","white","red","gray","silver"])
 
        c1, c2 = st.columns(2)
        vehicle_price    = c1.number_input("Harga Kendaraan (USD)", 0.0, 200000.0, 30000.0, step=500.0)
        vehicle_weight   = c2.number_input("Berat Kendaraan (kg)", 0.0, 100000.0, 15000.0, step=100.0)
 
        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)
 
    if submitted:
        input_dict = {
            "claim_number":"","age_of_driver": age_of_driver, "gender": gender,
            "marital_status": marital_status, "safty_rating": safty_rating,
            "annual_income": annual_income, "high_education_ind": high_education_ind,
            "address_change_ind": address_change_ind, "living_status": living_status,
            "zip_code": zip_code, "claim_date": claim_date,
            "claim_day_of_week": claim_day_of_week, "accident_site": accident_site,
            "past_num_of_claims": past_num_of_claims, "witness_present_ind": witness_present_ind,
            "liab_prct": liab_prct, "channel": "Broker",
            "policy_report_filed_ind": policy_report_filed_ind,
            "claim_est_payout": claim_est_payout, "age_of_vehicle": age_of_vehicle,
            "vehicle_category": vehicle_category, "vehicle_price": vehicle_price,
            "vehicle_color": vehicle_color, "vehicle_weight": vehicle_weight,
        }
        raw_df   = pd.DataFrame([input_dict])
        prob     = predict([input_dict], proba=True)[0]
        level, icon, rec = risk_level(prob)

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilitas Fraud", f"{prob*100:.1f}%")
        m2.metric("Level Risiko", f"{icon} {level}")
        m3.metric("Confidence (Legitimate)", f"{(1-prob)*100:.1f}%")

        st.progress(float(prob))

        if prob < 0.25:
            st.success(f"**{icon} Risiko {level}** — {rec}")
        elif prob < 0.50:
            st.info(f"**{icon} Risiko {level}** — {rec}")
        elif prob < 0.75:
            st.warning(f"**{icon} Risiko {level}** — {rec}")
        else:
            st.error(f"**{icon} Risiko {level}** — {rec}")

        st.markdown("---")
        st.subheader("🔬 Explainable AI — Kontribusi Fitur")
        st.markdown(
            "Grafik berikut menunjukkan fitur mana yang **meningkatkan** (merah) "
            "atau **menurunkan** (biru) probabilitas fraud pada prediksi ini."
        )
        fig = shap_bar_chart(raw_df)
        st.pyplot(fig)
        plt.close()

        with st.expander("📋 Lihat Data Input"):
            st.dataframe(raw_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "batch":

    # ── Sub-page: Detail row ───────────────────────────────────────────
    if st.session_state.detail_idx is not None:
        idx = st.session_state.detail_idx
        result_df = st.session_state.batch_result
        row = result_df.iloc[idx]

        if st.button("← Kembali ke Tabel Batch"):
            st.session_state.detail_idx = None
            st.rerun()

        prob  = row["fraud_probability"]
        level, icon, rec = risk_level(prob)

        st.title(f"Detail Klaim #{int(row.get('claim_number', idx+1))}")
        st.markdown("---")

        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilitas Fraud", f"{prob*100:.1f}%")
        m2.metric("Level Risiko", f"{icon} {level}")
        m3.metric("Confidence (Legitimate)", f"{(1-prob)*100:.1f}%")

        st.progress(float(prob))

        if prob < 0.25:
            st.success(f"**{icon} Risiko {level}** — {rec}")
        elif prob < 0.50:
            st.info(f"**{icon} Risiko {level}** — {rec}")
        elif prob < 0.75:
            st.warning(f"**{icon} Risiko {level}** — {rec}")
        else:
            st.error(f"**{icon} Risiko {level}** — {rec}")

        st.markdown("---")
        st.subheader("🔬 Explainable AI — Kontribusi Fitur")
        st.markdown(
            "Grafik berikut menunjukkan fitur mana yang **meningkatkan** (merah) "
            "atau **menurunkan** (biru) probabilitas fraud pada prediksi ini."
        )

        drop_cols = ["fraud_probability", "risk_level", "risk_icon", "recommendation"]
        if "fraud" in result_df.columns:
            drop_cols.append("fraud")
        raw_row = result_df.drop(columns=drop_cols).iloc[[idx]]
       
        fig = shap_bar_chart(raw_row, title=f"SHAP — Klaim #{int(row.get('claim_number', idx+1))}")
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("📋 Data Klaim")
        st.dataframe(raw_row.T.rename(columns={raw_row.index[0]: "Nilai"}), use_container_width=True)

    # ── Sub-page: Batch table ──────────────────────────────────────────
    else:
        st.title("📂 Batch Prediction")
        st.markdown(
            "Upload file CSV berisi data klaim. Sistem akan memprediksi probabilitas fraud "
            "untuk setiap baris dan menyajikannya dalam tabel interaktif."
        )
        st.markdown("---")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            raw_df = pd.read_csv(uploaded)
            st.success(f"File berhasil diupload — {len(raw_df):,} baris ditemukan.")

            with st.spinner("Memproses prediksi..."):
                probs   = predict(raw_df.to_dict(orient='records'), proba=True)

            result_df = raw_df.copy()
            result_df["fraud_probability"] = probs
            result_df["risk_level"]  = [risk_level(p)[0] for p in probs]
            result_df["risk_icon"]   = [risk_level(p)[1] for p in probs]
            result_df["recommendation"] = [risk_level(p)[2] for p in probs]

            st.session_state.batch_df     = raw_df
            st.session_state.batch_result = result_df

        if st.session_state.batch_result is not None:
            result_df = st.session_state.batch_result
            probs     = result_df["fraud_probability"].values

            st.markdown("---")
            st.subheader("📊 Ringkasan")

            total   = len(result_df)
            high    = (probs >= 0.50).sum()
            med     = ((probs >= 0.25) & (probs < 0.50)).sum()
            low     = (probs < 0.25).sum()
            avg_p   = probs.mean()

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Klaim",       f"{total:,}")
            m2.metric("🔴 Sangat Tinggi / Tinggi", f"{high:,}")
            m3.metric("🟡 Sedang",         f"{med:,}")
            m4.metric("🟢 Rendah",         f"{low:,}")
            m5.metric("Rata-rata Prob.",   f"{avg_p*100:.1f}%")

            st.markdown("---")
            st.subheader("📋 Tabel Hasil Prediksi")
            st.markdown("Klik tombol **Detail** pada baris untuk melihat penjelasan lengkap.")

            # Filter
            fc1, fc2 = st.columns([2, 1])
            filter_level = fc1.multiselect(
                "Filter Level Risiko",
                ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"],
                default=["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
            )
            sort_desc = fc2.checkbox("Urutkan probabilitas tertinggi", value=True)

            filtered = result_df[result_df["risk_level"].isin(filter_level)].copy()
            if sort_desc:
                filtered = filtered.sort_values("fraud_probability", ascending=False)
            filtered = filtered.reset_index(drop=True).head(10)

            display_cols = ["claim_number", "age_of_driver", "gender", "accident_site",
                            "channel", "fraud_probability", "risk_icon", "risk_level"]
            display_cols = [c for c in display_cols if c in filtered.columns]

            # Render table with detail buttons
            header_cols = st.columns([1, 1, 1, 1, 1, 1.5, 0.5, 1.2, 1])
            headers = ["Claim #", "Usia", "Gender", "Lokasi", "Channel", "Prob. Fraud", "Icon", "Level", "Aksi"]
            for col, h in zip(header_cols, headers):
                col.markdown(f"**{h}**")

            st.markdown("---")

            for i, (_, row) in enumerate(filtered.iterrows()):
                cols = st.columns([1, 1, 1, 1, 1, 1.5, 0.5, 1.2, 1])
                cols[0].write(int(row.get("claim_number", i+1)))
                cols[1].write(int(row.get("age_of_driver", "-")))
                cols[2].write(row.get("gender", "-"))
                cols[3].write(row.get("accident_site", "-"))
                cols[4].write(row.get("channel", "-"))
                cols[5].write(f"{row['fraud_probability']*100:.1f}%")
                cols[6].write(row["risk_icon"])
                cols[7].write(row["risk_level"])

                # Find original index
                original_idx = result_df.index[
                    result_df["claim_number"] == row.get("claim_number", -1)
                ].tolist()
                orig_i = original_idx[0] if original_idx else i

                if cols[8].button("Detail", key=f"detail_{i}_{orig_i}"):
                    st.session_state.detail_idx = orig_i
                    st.rerun()

            st.markdown("---")

            # Download
            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil CSV",
                data=csv_out,
                file_name="fraud_prediction_result.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Belum ada data. Silakan upload file CSV di atas.")
            with st.expander("📌 Format CSV yang diharapkan"):
                st.markdown(
                    "File CSV harus memiliki kolom berikut:\n\n"
                    "`claim_number, age_of_driver, gender, marital_status, safty_rating, "
                    "annual_income, high_education_ind, address_change_ind, living_status, "
                    "zip_code, claim_date, claim_day_of_week, accident_site, past_num_of_claims, "
                    "witness_present_ind, liab_prct, channel, policy_report_filed_ind, "
                    "claim_est_payout, age_of_vehicle, vehicle_category, vehicle_price, "
                    "vehicle_color, vehicle_weight`"
                )