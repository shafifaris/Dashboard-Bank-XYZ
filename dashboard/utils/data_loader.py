import os
import pandas as pd
import numpy as np
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def data_path(filename):
    return os.path.join(DATA_DIR, filename)

@st.cache_data
def load_master():
    return pd.read_csv(data_path('processed_bankxyz.csv'))

@st.cache_data
def load_branch():
    return pd.read_csv(data_path('agg_branch.csv'))

@st.cache_data
def load_provinsi():
    return pd.read_csv(data_path('agg_provinsi.csv'))

@st.cache_data
def load_gender():
    return pd.read_csv(data_path('agg_gender.csv'))

@st.cache_data
def load_panel():
    return pd.read_csv(data_path('agg_panel.csv'))

@st.cache_data
def load_usia():
    return pd.read_csv(data_path('agg_usia.csv'))

@st.cache_data
def load_segmen():
    return pd.read_csv(data_path('agg_segmen.csv'))

@st.cache_data
def load_ipa():
    return pd.read_csv(data_path('ipa_matrix.csv'))

@st.cache_data
def load_ipa_panel():
    return pd.read_csv(data_path('ipa_per_panel.csv'))

@st.cache_data
def load_emotion():
    return pd.read_csv(data_path('emotion_summary.csv'))

@st.cache_data
def load_emotion_panel():
    return pd.read_csv(data_path('emotion_per_panel.csv'))

@st.cache_data
def load_emotion_segmen():
    return pd.read_csv(data_path('emotion_per_segmen.csv'))

@st.cache_data
def load_brand():
    return pd.read_csv(data_path('brand_perception.csv'))

@st.cache_data
def load_competitor():
    path = data_path('competitor_benchmark.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_comp_provinsi():
    return pd.read_csv(data_path('comp_nps_per_provinsi.csv'))

@st.cache_data
def load_nps_competitor():
    return pd.read_csv(data_path('nps_competitor.csv'))

@st.cache_data
def load_driver():
    return pd.read_csv(data_path('driver_analysis.csv'))

@st.cache_data
def load_overall():
    return pd.read_csv(data_path('overall_satisfaction.csv'))

@st.cache_data
def load_segmen_profile():
    return pd.read_csv(data_path('segmen_profile.csv'))

@st.cache_data
def load_waiting():
    path = data_path('waiting_time.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_digitalisasi():
    return pd.read_csv(data_path('digitalisasi.csv'))

@st.cache_data
def load_nps_gender_panel():
    return pd.read_csv(data_path('nps_gender_panel.csv'))

@st.cache_data
def load_nps_usia_panel():
    return pd.read_csv(data_path('nps_usia_panel.csv'))

@st.cache_data
def load_nps_prov_panel():
    return pd.read_csv(data_path('nps_prov_panel.csv'))

@st.cache_data
def load_switching():
    path = data_path('switching_analysis.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_multibank():
    path = data_path('multibank_analysis.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_geojson():
    import json
    path = data_path('indonesia_provinces.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def nps_score(series):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return np.nan
    return float(round(((s >= 9).sum() - (s <= 6).sum()) / len(s) * 100, 1))

def nps_color(score):
    if pd.isna(score):
        return "#94a3b8"
    if score >= 50:
        return "#22c55e"
    if score >= 20:
        return "#f59e0b"
    if score >= 0:
        return "#f97316"
    return "#ef4444"

def get_branch_status(nps):
    if pd.isna(nps):
        return "Unknown"
    if nps >= 50:
        return "Healthy"
    if nps >= 20:
        return "Warning"
    return "Critical"

def apply_filters(df, provinsi=None, panel=None, kota=None):
    """
    Filter dataframe berdasarkan provinsi, panel, dan kota/kab.
    Otomatis mendeteksi nama kolom yang relevan.
    """
    filtered = df.copy()

    # ── Deteksi kolom provinsi ─────────────────────────────────
    prov_candidates = ['provinsi', 'PROVINSI', 'province', 'PROV', 'prov']
    prov_col = next((c for c in prov_candidates if c in df.columns), None)

    # ── Deteksi kolom panel ────────────────────────────────────
    panel_candidates = ['panel', 'PANEL', 'Panel']
    panel_col = next((c for c in panel_candidates if c in df.columns), None)

    # ── Deteksi kolom kota/kab ─────────────────────────────────
    kota_candidates = ['kota', 'KOTA', 'kabkota', 'KABKOTA', 'kab_kota',
                       'kota_kab', 'city', 'CITY', 'kab', 'KAB']
    kota_col = next((c for c in kota_candidates if c in df.columns), None)

    # ── Terapkan filter provinsi ───────────────────────────────
    if provinsi and provinsi not in ('All Provinces', '', None) and prov_col:
        filtered = filtered[filtered[prov_col] == provinsi]

    # ── Terapkan filter panel ──────────────────────────────────
    if panel and panel not in ('All Panels', '', None) and panel_col:
        filtered = filtered[filtered[panel_col] == panel]

    # ── Terapkan filter kota/kab ───────────────────────────────
    if kota and kota not in ('All Cities', '', None) and kota_col:
        filtered = filtered[filtered[kota_col] == kota]

    return filtered
