import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Analisis Data Klasifikasi 9 Jenis Sampah",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.dashboard-header {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%);
    padding: 2rem 2.5rem 1.8rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(27,94,32,0.35);
}
.dashboard-header h1 { color: #fff; font-size: 1.75rem; font-weight: 700; margin: 0; }
.dashboard-header p  { color: #C8E6C9; margin: 0.4rem 0 0; font-size: 0.95rem; }

.metric-card {
    background: #fff;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 5px solid;
    margin-bottom: 0.5rem;
}
.metric-card .label { color: #6B7280; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing:.05em; }
.metric-card .value { color: #111827; font-size: 2rem; font-weight: 700; margin: 0.2rem 0 0; }
.metric-card .sub   { color: #9CA3AF; font-size: 0.78rem; margin-top: 0.2rem; }

.section-title {
    font-size: 1.1rem; font-weight: 600; color: #1B5E20;
    border-bottom: 3px solid #A5D6A7;
    padding-bottom: 0.4rem; margin-bottom: 1rem;
}

.insight-box {
    background: #F0FDF4;
    border-left: 4px solid #4CAF50;
    padding: 0.8rem 1rem;
    border-radius: 0 10px 10px 0;
    margin: 0.5rem 0 1rem;
    font-size: 0.88rem;
    color: #1B5E20;
}
.insight-box.warning {
    background: #FFFDE7;
    border-left-color: #F9A825;
    color: #5D4037;
}

[data-testid="stSidebar"] { background: #F9FBF9; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA ASLI
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("eda_result.csv")

    WASTE_GROUP = {
        "Kaca": "Recyclable", "Kardus": "Recyclable", "Kertas": "Recyclable",
        "Logam": "Recyclable", "Plastik": "Recyclable", "recyclable": "Recyclable",
        "organic": "Organic", "hazardous": "Hazardous", "Residu": "Residu"
    }

    # Feature Engineering
    df["aspect_ratio"]   = (df["width"] / df["height"]).round(4)
    df["pixel_area"]     = df["width"] * df["height"]

    def cat_res(a):
        if a < 50_000:   return "Low"
        elif a <= 500_000: return "Medium"
        return "High"

    def cat_size(kb):
        if kb < 5:     return "Small"
        elif kb <= 500: return "Normal"
        return "Large"

    def flag_quality(row):
        r, s = row["resolution_category"], row["file_size_category"]
        if r == "Low" or s == "Small":              return "Poor"
        if r in ["Medium","High"] and s in ["Normal","Large"]: return "Good"
        return "Review"

    df["resolution_category"] = df["pixel_area"].apply(cat_res)
    df["file_size_category"]  = df["file_size_kb"].apply(cat_size)
    df["waste_group"]         = df["class"].map(WASTE_GROUP)

    color_map = {
        "RGB": "RGB", "RGBA": "RGBA_needs_conversion",
        "L": "Grayscale_needs_conversion", "P": "Palette_needs_conversion"
    }
    df["color_mode_label"] = df["mode"].map(color_map).fillna("Unknown")
    df["image_quality_flag"] = df.apply(flag_quality, axis=1)

    mean_count = df["class"].value_counts().mean()
    threshold  = mean_count * 0.8
    need_aug   = df["class"].value_counts()[df["class"].value_counts() < threshold].index.tolist()
    df["needs_augmentation"] = df["class"].apply(lambda c: "Yes" if c in need_aug else "No")

    return df, need_aug, mean_count, threshold

df, need_aug, mean_count, threshold = load_data()

COLOR_PALETTE = {
    "hazardous": "#EF5350", "Kaca": "#42A5F5", "Kardus": "#8D6E63",
    "Kertas": "#FFA726", "Logam": "#78909C", "organic": "#66BB6A",
    "Plastik": "#EC407A", "recyclable": "#26A69A", "Residu": "#9E9E9E"
}
WG_COLOR = {
    "Recyclable": "#42A5F5", "Organic": "#66BB6A",
    "Hazardous": "#EF5350", "Residu": "#9E9E9E"
}

CLASSES = sorted(df["class"].unique().tolist())

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂️ Filter Data")
    st.markdown("---")

    sel_class = st.multiselect("Pilih Kelas Sampah", options=CLASSES, default=CLASSES)
    sel_wg    = st.multiselect(
        "Pilih Waste Group",
        options=["Recyclable", "Organic", "Hazardous", "Residu"],
        default=["Recyclable", "Organic", "Hazardous", "Residu"]
    )
    sel_quality = st.multiselect(
        "Filter Kualitas Gambar",
        options=["Good", "Review", "Poor"],
        default=["Good", "Review", "Poor"]
    )

    st.markdown("---")
    st.markdown("#### ℹ️ Tentang Dataset")
    st.markdown(f"""
    - **Total gambar:** {len(df):,}
    - **Total kelas:** {df['class'].nunique()}
    - **Resolusi:** semua 224 × 224 px
    - **Format:** semua `.jpg` (RGB)
    """)
    st.markdown("---")
    st.caption("Data bersumber dari `eda_result.csv`")

mask = (
    df["class"].isin(sel_class) &
    df["waste_group"].isin(sel_wg) &
    df["image_quality_flag"].isin(sel_quality)
)
dff = df[mask]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
  <h1>♻️ Dashboard Analisis Data Klasifikasi 9 Jenis Sampah</h1>
  <p>Eksplorasi dataset gambar sampah — distribusi kelas, kualitas gambar, dan hasil feature engineering</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

good_pct  = round(len(dff[dff["image_quality_flag"] == "Good"]) / max(len(dff),1) * 100, 1)
aug_cls_n = len([c for c in dff["class"].unique() if c in need_aug])

with c1:
    st.markdown(f"""<div class="metric-card" style="border-color:#2E7D32;">
        <div class="label">Total Gambar</div>
        <div class="value">{len(dff):,}</div>
        <div class="sub">Setelah filter aktif</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card" style="border-color:#1565C0;">
        <div class="label">Kelas Aktif</div>
        <div class="value">{dff['class'].nunique()}</div>
        <div class="sub">Dari 9 kelas tersedia</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card" style="border-color:#4CAF50;">
        <div class="label">Kualitas Good</div>
        <div class="value">{good_pct}%</div>
        <div class="sub">Siap untuk training</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card" style="border-color:#FF7043;">
        <div class="label">Perlu Augmentasi</div>
        <div class="value">{aug_cls_n} kelas</div>
        <div class="sub">Di bawah threshold 80%</div>
    </div>""", unsafe_allow_html=True)
with c5:
    imbalance = round(dff["class"].value_counts().max() / max(dff["class"].value_counts().min(),1), 2)
    st.markdown(f"""<div class="metric-card" style="border-color:#7B1FA2;">
        <div class="label">Rasio Imbalance</div>
        <div class="value">{imbalance}×</div>
        <div class="sub">Maks / Min kelas</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA — Distribusi Data",
    "🔍 EDA — Ukuran & Format",
    "🛠️ Feature Engineering",
    "📋 Ringkasan & Rekomendasi"
])

# ══════════════════════════
# TAB 1
# ══════════════════════════
with tab1:
    st.markdown('<div class="section-title">Distribusi Jumlah Gambar per Kelas</div>', unsafe_allow_html=True)

    cc = dff["class"].value_counts().reset_index()
    cc.columns = ["Kelas", "Jumlah"]

    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig_bar = px.bar(
            cc.sort_values("Jumlah", ascending=True),
            x="Jumlah", y="Kelas", orientation="h",
            color="Kelas", color_discrete_map=COLOR_PALETTE,
            text="Jumlah",
            title="Jumlah Gambar per Kelas Sampah",
            labels={"Jumlah": "Jumlah Gambar", "Kelas": ""}
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            showlegend=False, height=420, plot_bgcolor="white",
            paper_bgcolor="white", xaxis=dict(gridcolor="#F3F4F6"),
            margin=dict(l=10, r=40, t=50, b=10), title_font_size=14
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_thresh = go.Figure()
        for _, row in cc.iterrows():
            fig_thresh.add_trace(go.Bar(
                x=[row["Kelas"]], y=[row["Jumlah"]],
                marker_color=COLOR_PALETTE.get(row["Kelas"], "#9E9E9E"),
                name=row["Kelas"], showlegend=False,
                text=[row["Jumlah"]], textposition="outside"
            ))
        fig_thresh.add_hline(y=mean_count, line_dash="dash", line_color="#E53935",
                              annotation_text=f"Rata-rata: {mean_count:.0f}",
                              annotation_position="top right")
        fig_thresh.add_hline(y=threshold, line_dash="dot", line_color="#FB8C00",
                              annotation_text=f"Threshold 80%: {threshold:.0f}",
                              annotation_position="bottom right")
        fig_thresh.update_layout(
            title="Threshold Augmentasi", height=420,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F3F4F6"),
            xaxis_tickangle=-40, margin=dict(l=10, r=10, t=50, b=10),
            title_font_size=14
        )
        st.plotly_chart(fig_thresh, use_container_width=True)

    min_cls = cc.loc[cc["Jumlah"].idxmin()]
    max_cls = cc.loc[cc["Jumlah"].idxmax()]
    st.markdown(f"""
    <div class="insight-box">
        📌 <b>Class Imbalance:</b> Kelas terbanyak adalah <b>{max_cls['Kelas']}</b>
        ({max_cls['Jumlah']:,} gambar) dan paling sedikit <b>{min_cls['Kelas']}</b>
        ({min_cls['Jumlah']:,} gambar). Rasio imbalance = <b>{round(max_cls['Jumlah']/min_cls['Jumlah'],2)}×</b>.
    </div>
    <div class="insight-box warning">
        ⚠️ Kelas perlu augmentasi (di bawah threshold {threshold:.0f} gambar):
        <b>{', '.join(need_aug) if need_aug else 'Tidak ada'}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distribusi Waste Group</div>', unsafe_allow_html=True)

    wg_count = dff["waste_group"].value_counts().reset_index()
    wg_count.columns = ["Waste Group", "Jumlah"]

    col_c, col_d = st.columns(2)
    with col_c:
        fig_pie = px.pie(
            wg_count, names="Waste Group", values="Jumlah",
            color="Waste Group", color_discrete_map=WG_COLOR,
            title="Proporsi Waste Group", hole=0.45
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(height=380, paper_bgcolor="white",
                               margin=dict(t=60,b=10), showlegend=False, title_font_size=14)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_d:
        per_cls_wg = dff.groupby(["waste_group","class"]).size().reset_index(name="Jumlah")
        fig_sun = px.sunburst(
            per_cls_wg, path=["waste_group","class"], values="Jumlah",
            color="waste_group", color_discrete_map=WG_COLOR,
            title="Hierarki Waste Group → Kelas"
        )
        fig_sun.update_layout(height=380, paper_bgcolor="white",
                               margin=dict(t=60,b=10), title_font_size=14)
        st.plotly_chart(fig_sun, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        📌 <b>Recyclable</b> mendominasi dataset karena mencakup 6 kelas material sekaligus
        (Kaca, Kardus, Kertas, Logam, Plastik, recyclable). Pengelompokan ini
        berguna untuk model hierarkis <i>coarse-to-fine classification</i>.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════
# TAB 2
# ══════════════════════════
with tab2:
    st.markdown('<div class="section-title">Ukuran Gambar</div>', unsafe_allow_html=True)

    # Karena semua gambar 224x224, tampilkan info ini dengan jelas
    st.markdown("""
    <div class="insight-box">
        ✅ <b>Semua gambar sudah memiliki ukuran seragam 224 × 224 px.</b>
        Tidak diperlukan resize tambahan — dataset sudah siap masuk pipeline model seperti
        ResNet, EfficientNet, atau MobileNet tanpa preprocessing dimensi.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Distribusi file size per kelas
        fig_box = px.box(
            dff, x="class", y="file_size_kb",
            color="class", color_discrete_map=COLOR_PALETTE,
            title="Distribusi Ukuran File (KB) per Kelas",
            labels={"file_size_kb": "Ukuran File (KB)", "class": ""}
        )
        fig_box.update_layout(
            height=420, showlegend=False, plot_bgcolor="white",
            paper_bgcolor="white", xaxis_tickangle=-40,
            yaxis=dict(gridcolor="#F3F4F6"),
            margin=dict(t=60, b=10), title_font_size=14
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_b:
        fig_hist = px.histogram(
            dff, x="file_size_kb", nbins=40,
            color_discrete_sequence=["#42A5F5"],
            title="Distribusi Ukuran File (KB) — Keseluruhan",
            labels={"file_size_kb": "Ukuran File (KB)", "count": "Jumlah Gambar"}
        )
        fig_hist.add_vline(x=5, line_dash="dot", line_color="#E53935",
                            annotation_text="5 KB (batas Small)")
        fig_hist.update_layout(
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F3F4F6"),
            margin=dict(t=60, b=10), title_font_size=14
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Format File & Color Mode</div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        ext_cnt = dff["extension"].value_counts().reset_index()
        ext_cnt.columns = ["Format", "Jumlah"]
        fig_ext = px.bar(
            ext_cnt, x="Format", y="Jumlah",
            color_discrete_sequence=["#42A5F5"],
            text="Jumlah", title="Distribusi Format File",
            labels={"Format": "Ekstensi", "Jumlah": "Jumlah Gambar"}
        )
        fig_ext.update_traces(textposition="outside")
        fig_ext.update_layout(
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F3F4F6"),
            margin=dict(t=50, b=10), title_font_size=13
        )
        st.plotly_chart(fig_ext, use_container_width=True)

    with col_d:
        mode_cnt = dff["mode"].value_counts().reset_index()
        mode_cnt.columns = ["Mode", "Jumlah"]
        fig_mode = px.bar(
            mode_cnt, x="Mode", y="Jumlah",
            color_discrete_sequence=["#66BB6A"],
            text="Jumlah", title="Distribusi Color Mode",
            labels={"Mode": "Color Mode", "Jumlah": "Jumlah Gambar"}
        )
        fig_mode.update_traces(textposition="outside")
        fig_mode.update_layout(
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F3F4F6"),
            margin=dict(t=50, b=10), title_font_size=13
        )
        st.plotly_chart(fig_mode, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        ✅ <b>Seluruh gambar berformat .jpg dan berwarna RGB</b> — tidak ada konversi format
        maupun mode warna yang diperlukan sebelum proses training.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Statistik Deskriptif Ukuran File per Kelas</div>', unsafe_allow_html=True)

    stats = dff.groupby("class").agg(
        Jumlah=("file_size_kb", "count"),
        Min_KB=("file_size_kb", "min"),
        Mean_KB=("file_size_kb", "mean"),
        Median_KB=("file_size_kb", "median"),
        Max_KB=("file_size_kb", "max"),
        Std_KB=("file_size_kb", "std"),
    ).round(2).reset_index()
    stats.columns = ["Kelas","Jumlah","Min (KB)","Rata-rata (KB)","Median (KB)","Maks (KB)","Std (KB)"]

    st.dataframe(
        stats.style
            .background_gradient(subset=["Jumlah"], cmap="Greens")
            .background_gradient(subset=["Rata-rata (KB)"], cmap="Blues")
            .format({c: "{:.2f}" for c in stats.columns if c != "Kelas" and c != "Jumlah"}),
        use_container_width=True, hide_index=True
    )

# ══════════════════════════
# TAB 3
# ══════════════════════════
with tab3:
    st.markdown('<div class="section-title">7 Fitur Baru dari Feature Engineering</div>', unsafe_allow_html=True)

    features = [
        ("aspect_ratio",        "#42A5F5", "📐", "Rasio lebar / tinggi gambar (kuantitatif)"),
        ("resolution_category", "#66BB6A", "🖼️", "Low / Medium / High berdasarkan luas piksel"),
        ("file_size_category",  "#FFA726", "💾", "Small / Normal / Large berdasarkan KB"),
        ("color_mode_label",    "#AB47BC", "🎨", "Label konversi mode warna"),
        ("waste_group",         "#26A69A", "🗑️", "Recyclable / Organic / Hazardous / Residu"),
        ("image_quality_flag",  "#EF5350", "🚦", "Good / Review / Poor"),
        ("needs_augmentation",  "#78909C", "🔄", "Flag kelas butuh augmentasi"),
    ]
    cols = st.columns(4)
    for i, (name, color, icon, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:1rem;
                        box-shadow:0 2px 10px rgba(0,0,0,0.08);
                        border-top:4px solid {color};margin-bottom:0.8rem;">
                <div style="font-size:1.5rem">{icon}</div>
                <div style="font-weight:600;font-size:0.85rem;color:#111;margin:.3rem 0 .2rem">{name}</div>
                <div style="font-size:0.75rem;color:#6B7280">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Resolution & File Size Category</div>', unsafe_allow_html=True)

        res_dist = dff["resolution_category"].value_counts().reset_index()
        res_dist.columns = ["Kategori", "Jumlah"]
        res_dist["Tipe"] = "Resolution"

        fs_dist = dff["file_size_category"].value_counts().reset_index()
        fs_dist.columns = ["Kategori", "Jumlah"]
        fs_dist["Tipe"] = "File Size"

        combined = pd.concat([res_dist, fs_dist])
        fig_cat = px.bar(
            combined, x="Kategori", y="Jumlah", color="Tipe",
            barmode="group",
            color_discrete_sequence=["#42A5F5", "#FFA726"],
            text="Jumlah",
            title="Distribusi Resolution & File Size Category",
            labels={"Jumlah": "Jumlah Gambar", "Kategori": ""}
        )
        fig_cat.update_traces(textposition="outside")
        fig_cat.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F3F4F6"),
            margin=dict(t=60, b=10), title_font_size=13
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Image Quality Flag</div>', unsafe_allow_html=True)

        qual_data = dff["image_quality_flag"].value_counts().reset_index()
        qual_data.columns = ["Quality", "Jumlah"]
        fig_qual = px.pie(
            qual_data, names="Quality", values="Jumlah",
            color="Quality",
            color_discrete_map={"Good": "#66BB6A", "Review": "#FFA726", "Poor": "#EF5350"},
            title="Proporsi Image Quality Flag", hole=0.45
        )
        fig_qual.update_traces(textinfo="percent+label+value")
        fig_qual.update_layout(
            height=380, paper_bgcolor="white",
            margin=dict(t=60, b=10), showlegend=False, title_font_size=13
        )
        st.plotly_chart(fig_qual, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Needs Augmentation per Kelas</div>', unsafe_allow_html=True)

        aug_df = dff.groupby("class").agg(
            Jumlah=("needs_augmentation", "count")
        ).reset_index()
        aug_df["Status"] = aug_df["class"].apply(
            lambda c: "🔴 Perlu Augmentasi" if c in need_aug else "🟢 Sudah Cukup"
        )
        fig_aug = px.bar(
            aug_df.sort_values("Jumlah"),
            x="Jumlah", y="class", orientation="h",
            color="Status",
            color_discrete_map={"🔴 Perlu Augmentasi": "#EF5350", "🟢 Sudah Cukup": "#66BB6A"},
            text="Jumlah",
            labels={"class": "", "Jumlah": "Jumlah Gambar"},
            title="Status Augmentasi per Kelas"
        )
        fig_aug.add_vline(x=threshold, line_dash="dot", line_color="#FB8C00",
                           annotation_text=f"Threshold: {threshold:.0f}")
        fig_aug.update_traces(textposition="outside")
        fig_aug.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#F3F4F6"), showlegend=True,
            margin=dict(t=60, b=10), title_font_size=13,
            legend=dict(orientation="h", y=-0.3)
        )
        st.plotly_chart(fig_aug, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Waste Group Distribution</div>', unsafe_allow_html=True)

        wg_class = dff.groupby(["waste_group","class"]).size().reset_index(name="Jumlah")
        fig_wg = px.bar(
            wg_class, x="waste_group", y="Jumlah", color="class",
            color_discrete_map=COLOR_PALETTE,
            text="Jumlah", barmode="stack",
            title="Komposisi Kelas dalam Tiap Waste Group",
            labels={"waste_group": "Waste Group", "Jumlah": "Jumlah Gambar"}
        )
        fig_wg.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F3F4F6"),
            margin=dict(t=60, b=10), title_font_size=13,
            legend=dict(title="Kelas", orientation="h", y=-0.4)
        )
        st.plotly_chart(fig_wg, use_container_width=True)

    # Cross-tab kualitas per kelas
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cross-Tab: Kualitas Gambar per Kelas</div>', unsafe_allow_html=True)

    cross = dff.groupby(["class","image_quality_flag"]).size().unstack(fill_value=0).reset_index()
    for col in ["Good","Review","Poor"]:
        if col not in cross.columns:
            cross[col] = 0
    cross["Total"] = cross.get("Good",0) + cross.get("Review",0) + cross.get("Poor",0)
    cross["% Good"] = (cross.get("Good",0) / cross["Total"] * 100).round(1)

    st.dataframe(
        cross[["class","Total","Good","Review","Poor","% Good"]]
            .rename(columns={"class":"Kelas"})
            .style.background_gradient(subset=["% Good"], cmap="Greens")
                  .background_gradient(subset=["Poor"], cmap="Reds"),
        use_container_width=True, hide_index=True
    )

# ══════════════════════════
# TAB 4
# ══════════════════════════
with tab4:
    st.markdown('<div class="section-title">📝 Ringkasan Temuan EDA</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🔵 Jawaban Business Questions")
        findings = [
            ("1. Distribusi Kelas",
             f"Dataset <b>tidak seimbang</b>. Kaca (452) mendominasi sementara Kertas & Logam (masing-masing 99) paling sedikit. Rasio imbalance mencapai <b>{round(452/99,1)}×</b>."),
            ("2. Variasi Ukuran Gambar",
             "Semua gambar sudah berukuran seragam <b>224 × 224 px</b> — tidak diperlukan standarisasi resolusi tambahan."),
            ("3. Distribusi Ukuran File",
             f"File berkisar antara {df['file_size_kb'].min():.1f}–{df['file_size_kb'].max():.1f} KB, rata-rata {df['file_size_kb'].mean():.1f} KB. Sebagian kecil gambar berukuran sangat kecil (<5 KB) berpotensi berkualitas rendah."),
            ("4. Representasi Visual",
             "Semua gambar RGB (.jpg). Setiap kelas memiliki variasi gambar, namun kelas minoritas perlu augmentasi untuk representasi yang lebih kaya."),
        ]
        for title, text in findings:
            st.markdown(f"""
            <div style="background:#fff;border-radius:10px;padding:0.9rem 1.1rem;
                        margin-bottom:0.6rem;box-shadow:0 1px 8px rgba(0,0,0,0.06);
                        border-left:4px solid #42A5F5;">
                <b style="color:#1565C0">{title}</b><br>
                <span style="font-size:0.86rem;color:#374151">{text}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("#### 🛠️ Fitur Baru yang Ditambahkan")
        fe_summary = [
            ("aspect_ratio",        "Numerik",   "Rasio geometri (w/h) gambar"),
            ("pixel_area",          "Numerik",   "Luas piksel (w × h)"),
            ("resolution_category", "Kategorik", "Low / Medium / High"),
            ("file_size_category",  "Kategorik", "Small / Normal / Large"),
            ("color_mode_label",    "Kategorik", "Label konversi mode warna"),
            ("waste_group",         "Kategorik", "Grup pengelolaan sampah"),
            ("image_quality_flag",  "Kategorik", "Good / Review / Poor"),
            ("needs_augmentation",  "Biner",     "Flag kelas perlu augmentasi"),
        ]
        fe_df = pd.DataFrame(fe_summary, columns=["Fitur","Tipe","Deskripsi"])
        st.dataframe(fe_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:#F0FDF4;border-radius:10px;padding:.8rem 1rem;margin-top:.5rem;font-size:.85rem;">
            <b>Total kolom:</b> 7 (original) + 8 (fitur baru) = <b>15 kolom</b><br>
            <b>Total baris:</b> <b>{len(df):,} gambar</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚀 Rekomendasi untuk Tahap Selanjutnya</div>', unsafe_allow_html=True)

    recs = [
        ("🔄", "Augmentasi Data", "#E8F5E9", "#2E7D32",
         f"Lakukan augmentasi (flip, rotasi, brightness jitter) pada kelas: <b>{', '.join(need_aug)}</b>. "
         f"Target minimal {threshold:.0f} gambar per kelas untuk mengatasi class imbalance."),
        ("🚦", "Filter Kualitas", "#FFF3E0", "#E65100",
         "Review manual gambar dengan flag <b>Poor</b> (ukuran file < 5 KB). "
         "Pertimbangkan untuk mengecualikannya dari dataset training."),
        ("🗑️", "Model Hierarkis", "#F3E5F5", "#6A1B9A",
         "Manfaatkan <b>waste_group</b> sebagai label coarse untuk model dua-tahap: "
         "pertama identifikasi grup (Recyclable/Organic/Hazardous/Residu), lalu kelas spesifik."),
        ("📊", "Arsitektur Model", "#E0F2F1", "#00695C",
         f"Dengan {len(df):,} gambar & resolusi 224×224 px, rekomendasikan <b>Transfer Learning</b> "
         "(EfficientNet-B0 / MobileNetV3) dengan fine-tuning. "
         "Gunakan class_weight untuk menangani imbalance saat training."),
    ]
    for icon, title, bg, color, text in recs:
        st.markdown(f"""
        <div style="background:{bg};border-radius:12px;padding:1rem 1.2rem;
                    margin-bottom:0.7rem;display:flex;gap:1rem;align-items:flex-start;">
            <span style="font-size:1.8rem;line-height:1">{icon}</span>
            <div>
                <b style="color:{color};font-size:0.95rem">{title}</b><br>
                <span style="font-size:0.86rem;color:#374151">{text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Summary KPI
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Statistik Dataset</div>', unsafe_allow_html=True)

    col_s = st.columns(4)
    metrics = [
        ("Total Gambar", f"{len(df):,}", "#42A5F5"),
        ("Total Kelas",  "9",            "#66BB6A"),
        ("Resolusi",     "224×224 px",   "#FFA726"),
        ("Format",       "100% JPG RGB", "#AB47BC"),
    ]
    for col, (label, val, color) in zip(col_s, metrics):
        with col:
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:1rem;
                        text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.08);
                        border-bottom:4px solid {color};">
                <div style="font-size:1.6rem;font-weight:700;color:{color}">{val}</div>
                <div style="font-size:0.78rem;color:#6B7280;margin-top:.3rem">{label}</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#9CA3AF;font-size:0.8rem;padding:1rem;border-top:1px solid #E5E7EB;">
    Dashboard Analisis Data Klasifikasi 9 Jenis Sampah
    &nbsp;|&nbsp; Data: eda_result.csv (2.003 gambar)
    &nbsp;|&nbsp; Referensi: SNI 3242:2008 & PerMenLHK No. 75 Tahun 2019
</div>
""", unsafe_allow_html=True)
