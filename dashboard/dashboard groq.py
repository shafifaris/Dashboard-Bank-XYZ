import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from typing import Dict, Any

# ==========================================
# 1. CONFIGURATION PARAMETERS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
GROQ_MODEL = "llama-3.3-70b-versatile"
PAGE_TITLE = "Bank XYZ Dashboard"

# ==========================================
# 2. PAGE SETUP & STYLING
# ==========================================
def setup_streamlit_page() -> None:
    """Menginisialisasi konfigurasi halaman dan CSS bawaan untuk Streamlit."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="collapsed")
    st.markdown("""<style>
      #root>div:first-child{margin:0;padding:0;}
      .stApp{margin:0;padding:0;}
      [data-testid="stHeader"]{display:none;}
      [data-testid="stSidebar"]{display:none;}
      [data-testid="stToolbar"]{display:none;}
      footer{display:none;} #MainMenu{display:none;}
      .stChatMessage { background-color: #F5F6FA !important; border-radius: 12px !important; border: 1px solid #E8ECF4 !important; }
    </style>""", unsafe_allow_html=True)


# ==========================================
# 3. DATA LAYER (ETL)
# ==========================================
@st.cache_data
def load_analytical_data() -> Dict[str, Any]:
    """Memuat semua file CSV hasil preprocessing dan merangkumnya ke dalam dictionary."""
    def read_csv_safe(filename: str) -> pd.DataFrame:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return pd.DataFrame()

    # Load master & base aggregates
    master           = read_csv_safe('processed_bankxyz.csv')
    branch           = read_csv_safe('agg_branch.csv')
    prov             = read_csv_safe('agg_provinsi.csv')
    gender           = read_csv_safe('agg_gender.csv')
    usia             = read_csv_safe('agg_usia.csv')
    panel            = read_csv_safe('agg_panel.csv')
    segmen           = read_csv_safe('agg_segmen.csv')
    ipa              = read_csv_safe('ipa_matrix.csv')
    emo              = read_csv_safe('emotion_summary.csv')
    brand            = read_csv_safe('brand_perception.csv')
    ovr              = read_csv_safe('overall_satisfaction.csv')
    comp             = read_csv_safe('competitor_benchmark.csv')
    nps_c            = read_csv_safe('nps_competitor.csv')
    sw               = read_csv_safe('switching_analysis.csv')
    digi             = read_csv_safe('digitalisasi.csv')
    
    # Load advanced analytics
    driver_analysis  = read_csv_safe('driver_analysis.csv')
    ipa_panel        = read_csv_safe('ipa_per_panel.csv')
    emo_segmen       = read_csv_safe('emotion_per_segmen.csv')
    emo_panel        = read_csv_safe('emotion_per_panel.csv')
    seg_profile      = read_csv_safe('segmen_profile.csv')
    wait_time        = read_csv_safe('waiting_time.csv')
    nps_g_panel      = read_csv_safe('nps_gender_panel.csv')
    nps_u_panel      = read_csv_safe('nps_usia_panel.csv')
    nps_p_panel      = read_csv_safe('nps_prov_panel.csv')
    comp_nps_prov    = read_csv_safe('comp_nps_per_provinsi.csv')

    # Helper function for calculating NPS on the fly
    def calculate_nps(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0: return 0.0
        return float(round(((s >= 9).sum() - (s <= 6).sum()) / len(s) * 100, 1))

    # Calculate Global KPIs
    gkpi = {
        'nps':          calculate_nps(master['nps_num']) if not master.empty else 0.0,
        'csi':          round(float(master['csi_num'].mean()), 2) if not master.empty else 0.0,
        'loyalty':      round(float(master['loyalty_num'].mean()), 2) if not master.empty else 0.0,
        'total':        int(len(master)),
        'promoters':    int((master['nps_num'] >= 9).sum()) if not master.empty else 0,
        'detractors':   int((master['nps_num'] <= 6).sum()) if not master.empty else 0,
        'passives':     int(((master['nps_num'] >= 7) & (master['nps_num'] <= 8)).sum()) if not master.empty else 0,
        'branches':     int(master['cabang'].nunique()) if not master.empty else 0,
        'provinces':    int(master['provinsi'].nunique()) if not master.empty else 0,
        'csi_pct_puas': round(float((master['csi_num'] >= 5).mean() * 100), 1) if not master.empty else 0.0,
    }

    # Prepare Branch & Province Rankings
    branch_s = branch.sort_values('nps_score', ascending=False).reset_index(drop=True) if not branch.empty else branch
    if not branch_s.empty: branch_s['rank'] = range(1, len(branch_s) + 1)
    prov_s   = prov.sort_values('nps_score', ascending=False).reset_index(drop=True) if not prov.empty else prov

    # Organize IPA Matrix data by category
    ipa_cats = {}
    if not ipa.empty:
        # Cek kolom yang tersedia untuk menghindari KeyError
        cols_to_extract = ['importance', 'performance', 'gap', 'kuadran']
        if 'atribut' in ipa.columns:
            cols_to_extract.insert(0, 'atribut')
        elif 'atribut_idx' in ipa.columns:
            cols_to_extract.insert(0, 'atribut_idx')

        for kat in ipa['kategori'].unique():
            sub = ipa[ipa['kategori'] == kat]
            records = sub[cols_to_extract].to_dict('records')
            
            # Memastikan frontend JavaScript tetap bisa membaca key 'atribut'
            for r in records:
                if 'atribut' not in r and 'atribut_idx' in r:
                    r['atribut'] = f"Atribut {r['atribut_idx']}"
                    
            ipa_cats[kat] = records

    # CSI Distribution
    csi_dist = {'scores': [], 'counts': []}
    if not master.empty and 'csi_num' in master.columns:
        csi_dist_raw = master['csi_num'].dropna().value_counts().sort_index()
        csi_dist = {'scores': [int(k) for k in csi_dist_raw.index], 'counts': [int(v) for v in csi_dist_raw.values]}

    # Panel Comparison Data
    panel_kpi = {}
    if not panel.empty:
        for _, row in panel.iterrows():
            panel_kpi[str(row['panel'])] = {
                'nps': float(row['nps_score']),
                'csi': round(float(row['csi_mean']), 2),
                'n': int(row['n'])
            }

    # Geospatial Lookups
    prov_kota   = master.groupby('provinsi')['kota'].unique().apply(sorted).to_dict() if not master.empty else {}
    kota_cabang = master.groupby('kota')['cabang'].unique().apply(sorted).to_dict() if not master.empty else {}

    return {
        'global_kpi': gkpi,
        'panel_kpi':  panel_kpi,
        'branch':     branch_s.to_dict('records'),
        'prov':       prov_s.to_dict('records'),
        'ipa_cats':   ipa_cats,
        'ovr_radar':  ovr[['kategori_layanan', 'mean_score']].to_dict('records') if not ovr.empty else [],
        'emo_pos':    emo[emo['tipe'] == 'positif'][['emosi', 'mean_score', 'pct_strong']].to_dict('records') if not emo.empty else [],
        'emo_neg':    emo[emo['tipe'] == 'negatif'][['emosi', 'mean_score', 'pct_strong']].to_dict('records') if not emo.empty else [],
        'csi_dist':   csi_dist,
        'segmen':     segmen.to_dict('records'),
        'gender':     gender.to_dict('records'),
        'usia':       usia.to_dict('records'),
        'comp':       comp.to_dict('records'),
        'nps_comp':   nps_c.to_dict('records'),
        'brand':      brand.to_dict('records'),
        'switch':     sw.to_dict('records'),
        'digi':       digi.to_dict('records'),
        'all_prov':   sorted(master['provinsi'].unique().tolist()) if not master.empty else [],
        'all_kota':   sorted(master['kota'].unique().tolist()) if not master.empty else [],
        'all_cabang': sorted(master['cabang'].unique().tolist()) if not master.empty else [],
        'all_panel':  sorted(master['panel'].unique().tolist()) if not master.empty else [],
        'prov_kota':  {k: v.tolist() if hasattr(v, 'tolist') else list(v) for k, v in prov_kota.items()},
        'kota_cabang':{k: v.tolist() if hasattr(v, 'tolist') else list(v) for k, v in kota_cabang.items()},
        'driver_analysis': driver_analysis.to_dict('records') if not driver_analysis.empty else [],
        'wait_time':  wait_time.to_dict('records') if not wait_time.empty else []
    }


# ==========================================
# 4. VIEW LAYER (FRONTEND HTML/JS)
# ==========================================
def get_dashboard_html(data_json: str) -> str:
    """Mengembalikan string HTML yang memuat seluruh UI/UX Dashboard kustom."""
    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {{
  --sidebar-w:220px; --sidebar-collapsed:64px;
  --orange:#E85D04; --orange-2:#F48C06; --orange-3:#FAA307;
  --bg:#F5F6FA; --card:#FFFFFF; --border:#E8ECF4;
  --text-1:#1A202C; --text-2:#4A5568; --text-3:#A0AEC0;
  --green:#10B981; --red:#EF4444; --blue:#3B82F6; --purple:#8B5CF6;
  --sidebar-bg:#1C0A00; --sidebar-hover:rgba(255,255,255,0.08);
  --sidebar-active:rgba(232,93,4,0.25); --radius:14px;
  --shadow:0 1px 8px rgba(0,0,0,0.07);
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text-1);height:100vh;overflow:hidden;display:flex;}}
#sidebar{{width:var(--sidebar-w);min-width:var(--sidebar-w);background:linear-gradient(170deg,#1C0A00 0%,#6B2200 55%,var(--orange) 100%);display:flex;flex-direction:column;transition:width .3s cubic-bezier(.4,0,.2,1),min-width .3s cubic-bezier(.4,0,.2,1);overflow:hidden;position:relative;z-index:100;box-shadow:2px 0 20px rgba(0,0,0,.15);}}
#sidebar.collapsed{{width:var(--sidebar-collapsed);min-width:var(--sidebar-collapsed);}}
.sidebar-logo{{padding:20px 16px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid rgba(255,255,255,.1);overflow:hidden;white-space:nowrap;}}
.sidebar-logo-icon{{font-size:26px;flex-shrink:0;width:32px;text-align:center;}}
.sidebar-logo-title{{font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:800;color:white;line-height:1.2;}}
.sidebar-logo-sub{{font-size:10px;color:rgba(255,255,255,.55);margin-top:2px;}}
.sidebar-section{{font-size:9px;font-weight:700;color:rgba(255,255,255,.35);text-transform:uppercase;letter-spacing:1.5px;padding:16px 18px 6px;white-space:nowrap;overflow:hidden;transition:opacity .2s;}}
#sidebar.collapsed .sidebar-section{{opacity:0;padding:12px 0 4px;}}
.nav-item{{display:flex;align-items:center;gap:12px;padding:10px 16px;margin:2px 8px;border-radius:10px;cursor:pointer;color:rgba(255,255,255,.7);font-size:13px;font-weight:500;white-space:nowrap;overflow:hidden;transition:background .2s,color .2s;user-select:none;}}
.nav-item:hover{{background:var(--sidebar-hover);color:white;}}
.nav-item.active{{background:var(--sidebar-active);color:white;font-weight:700;border:1px solid rgba(232,93,4,.4);}}
.nav-icon{{font-size:17px;flex-shrink:0;width:20px;text-align:center;}}
.nav-label{{transition:opacity .2s;font-size:13px;}}
#sidebar.collapsed .nav-label{{opacity:0;width:0;}}
.sidebar-toggle{{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);width:32px;height:32px;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.2);border-radius:8px;display:flex;align-items:center;justify-content:center;cursor:pointer;color:white;font-size:14px;transition:background .2s,transform .3s;user-select:none;z-index:10;}}
.sidebar-toggle:hover{{background:rgba(255,255,255,.25);}}
#sidebar.collapsed .sidebar-toggle{{transform:translateX(-50%) rotate(180deg);}}
#main{{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0;}}
#topbar{{background:var(--card);border-bottom:1px solid var(--border);padding:0 24px;height:56px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0;box-shadow:0 1px 4px rgba(0,0,0,.04);}}
.topbar-left{{display:flex;align-items:center;gap:12px;}}
.page-title{{font-family:'Plus Jakarta Sans',sans-serif;font-size:17px;font-weight:700;color:var(--text-1);}}
.page-sub{{font-size:12px;color:var(--text-3);}}
.topbar-filters{{display:flex;align-items:center;gap:10px;}}
.custom-select-wrap{{position:relative;display:inline-block;}}
.custom-select-btn{{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:6px 12px;font-size:12px;font-weight:500;color:var(--text-2);cursor:pointer;display:flex;align-items:center;gap:6px;white-space:nowrap;min-width:130px;user-select:none;transition:border-color .2s,box-shadow .2s;}}
.custom-select-btn:hover{{border-color:var(--orange);box-shadow:0 0 0 2px rgba(232,93,4,.12);}}
.custom-select-btn.open{{border-color:var(--orange);box-shadow:0 0 0 3px rgba(232,93,4,.15);}}
.select-label{{font-size:10px;color:var(--text-3);font-weight:600;text-transform:uppercase;letter-spacing:.8px;}}
.select-value{{font-size:12px;font-weight:600;color:var(--text-1);}}
.select-arrow{{margin-left:auto;color:var(--text-3);font-size:10px;transition:transform .2s;}}
.custom-select-btn.open .select-arrow{{transform:rotate(180deg);}}
.custom-dropdown{{position:absolute;top:calc(100% + 6px);left:0;min-width:200px;background:var(--card);border:1px solid var(--border);border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.12);z-index:1000;display:none;overflow:hidden;}}
.custom-dropdown.open{{display:block;}}
.dropdown-header{{padding:10px 14px 6px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}}
.dropdown-title{{font-size:11px;font-weight:700;color:var(--text-2);text-transform:uppercase;letter-spacing:.8px;}}
.dropdown-actions{{display:flex;gap:8px;}}
.dropdown-action{{font-size:11px;color:var(--orange);cursor:pointer;font-weight:600;}}
.dropdown-action:hover{{text-decoration:underline;}}
.dropdown-search{{padding:8px 14px;border-bottom:1px solid var(--border);}}
.dropdown-search input{{width:100%;border:1px solid var(--border);border-radius:6px;padding:5px 10px;font-size:12px;color:var(--text-1);outline:none;font-family:'Inter',sans-serif;}}
.dropdown-search input:focus{{border-color:var(--orange);}}
.dropdown-list{{max-height:200px;overflow-y:auto;padding:4px 0;}}
.dropdown-item{{display:flex;align-items:center;gap:10px;padding:7px 14px;cursor:pointer;font-size:12px;color:var(--text-2);transition:background .15s;}}
.dropdown-item:hover{{background:#FFF7ED;}}
.dropdown-item.checked{{color:var(--text-1);font-weight:500;}}
.custom-checkbox{{width:15px;height:15px;border:2px solid var(--border);border-radius:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;transition:all .15s;background:white;font-size:9px;}}
.dropdown-item.checked .custom-checkbox{{background:var(--orange);border-color:var(--orange);color:white;}}
#content{{flex:1;overflow-y:auto;padding:20px 24px;height: 100vh;}}
#content::-webkit-scrollbar{{width:5px;}}
#content::-webkit-scrollbar-track{{background:transparent;}}
#content::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px;}}
.kpi-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:22px;}}
.kpi-card{{background:var(--card);border-radius:var(--radius);padding:18px 20px;border:1px solid var(--border);box-shadow:var(--shadow);position:relative;overflow:hidden;}}
.kpi-card-label{{font-size:10px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}}
.kpi-card-value{{font-family:'Plus Jakarta Sans',sans-serif;font-size:28px;font-weight:800;color:var(--text-1);line-height:1;margin-bottom:8px;}}
.kpi-badge{{display:inline-flex;align-items:center;gap:4px;font-size:11px;font-weight:600;padding:3px 8px;border-radius:20px;}}
.badge-green{{color:var(--green);background:rgba(16,185,129,.1);}}
.badge-red{{color:var(--red);background:rgba(239,68,68,.1);}}
.badge-orange{{color:var(--orange);background:rgba(232,93,4,.1);}}
.badge-blue{{color:var(--blue);background:rgba(59,130,246,.1);}}
.badge-purple{{color:var(--purple);background:rgba(139,92,246,.1);}}
.chart-grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}}
.chart-grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px;}}
.chart-grid-13{{display:grid;grid-template-columns:1fr 2fr;gap:16px;margin-bottom:16px;}}
.chart-grid-31{{display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-bottom:16px;}}
.chart-card{{background:var(--card);border-radius:var(--radius);padding:18px 20px;border:1px solid var(--border);box-shadow:var(--shadow);}}
.chart-title{{font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:700;color:var(--text-1);margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;}}
.chart-subtitle{{font-size:11px;color:var(--text-3);font-weight:400;margin-top:2px;}}
.rank-list{{display:flex;flex-direction:column;gap:8px;}}
.rank-item{{display:flex;align-items:center;gap:12px;padding:10px 12px;background:var(--bg);border-radius:10px;border:1px solid var(--border);}}
.rank-num{{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;font-family:'Plus Jakarta Sans',sans-serif;flex-shrink:0;}}
.r1{{background:linear-gradient(135deg,#FFD700,#FFA500);color:white;}}
.r2{{background:linear-gradient(135deg,#C0C0C0,#909090);color:white;}}
.r3{{background:linear-gradient(135deg,#CD7F32,#A05A0A);color:white;}}
.rn{{background:#EDF2F7;color:var(--text-2);}}
.rank-name{{flex:1;font-size:12px;font-weight:600;color:var(--text-1);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.rank-score{{font-family:'Plus Jakarta Sans',sans-serif;font-size:16px;font-weight:800;color:var(--orange);}}
.rank-sub{{font-size:10px;color:var(--text-3);}}
.score-item{{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:var(--card);border:1px solid var(--border);border-radius:10px;margin-bottom:8px;}}
.score-left .score-metric{{font-size:13px;font-weight:600;color:var(--text-1);}}
.score-left .score-target{{font-size:11px;color:var(--text-3);margin-top:2px;}}
.score-right .score-val{{font-family:'Plus Jakarta Sans',sans-serif;font-size:16px;font-weight:800;text-align:right;}}
.score-right .score-status{{font-size:10px;font-weight:600;text-align:right;margin-top:2px;}}
.tab-row{{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap;}}
.tab-btn{{padding:5px 14px;border-radius:20px;font-size:11px;font-weight:600;cursor:pointer;border:1px solid var(--border);background:var(--bg);color:var(--text-2);transition:all .2s;}}
.tab-btn.active{{background:var(--orange);border-color:var(--orange);color:white;}}
.rec-card{{background:var(--card);border-radius:12px;padding:16px 18px;border:1px solid var(--border);border-left:4px solid var(--orange);box-shadow:var(--shadow);margin-bottom:12px;}}
.rec-num{{font-size:10px;font-weight:700;color:var(--orange);text-transform:uppercase;letter-spacing:1px;}}
.rec-title{{font-family:'Plus Jakarta Sans',sans-serif;font-size:13px;font-weight:700;color:var(--text-1);margin:4px 0;}}
.rec-body{{font-size:12px;color:var(--text-2);line-height:1.6;}}
.page{{display:none;}} .page.active{{display:block;}}
.plotly-notifier{{display:none!important;}}
</style>
</head>
<body>

<nav id="sidebar">
  <div class="sidebar-logo">
    <div class="sidebar-logo-icon">🏦</div>
    <div class="sidebar-logo-text">
      <div class="sidebar-logo-title">BankSurvey</div>
      <div class="sidebar-logo-sub">Customer Satisfaction 2024</div>
    </div>
  </div>
  <div class="sidebar-section">OVERVIEW</div>
  <div class="nav-item active" data-page="dashboard" onclick="navigate(this)">
    <span class="nav-icon">◆</span><span class="nav-label">Dashboard</span>
  </div>
  <div class="sidebar-section">ANALYSIS</div>
  <div class="nav-item" data-page="branch" onclick="navigate(this)">
    <span class="nav-icon">○</span><span class="nav-label">Branch Performance</span>
  </div>
  <div class="nav-item" data-page="touchpoint" onclick="navigate(this)">
    <span class="nav-icon">○</span><span class="nav-label">Touchpoint Analysis</span>
  </div>
  <div class="nav-item" data-page="demographics" onclick="navigate(this)">
    <span class="nav-icon">○</span><span class="nav-label">Demographics</span>
  </div>
  <div class="nav-item" data-page="competitor" onclick="navigate(this)">
    <span class="nav-icon">○</span><span class="nav-label">Competitor Benchmark</span>
  </div>
  <div class="sidebar-section">INSIGHT</div>
  <div class="nav-item" data-page="recommendation" onclick="navigate(this)">
    <span class="nav-icon">◆</span><span class="nav-label">Recommendation</span>
  </div>
  <div class="sidebar-toggle" onclick="toggleSidebar()">❮</div>
</nav>

<div id="main">
  <div id="topbar">
    <div class="topbar-left">
      <div>
        <div class="page-title" id="topbar-title">Dashboard Overview</div>
        <div class="page-sub">Bank XYZ · Customer Satisfaction Survey</div>
      </div>
    </div>
    <div class="topbar-filters" id="topbar-filters"></div>
  </div>

  <div id="content">

    <div class="page active" id="page-dashboard">
      <div class="kpi-grid" id="kpi-grid"></div>
      <div class="chart-grid-3">
        <div class="chart-card">
          <div class="chart-title">NPS Breakdown<div class="chart-subtitle">Promoter / Passive / Detractor</div></div>
          <div id="chart-nps-donut" style="height:220px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">NPS per Provinsi<div class="chart-subtitle">vs rata-rata keseluruhan</div></div>
          <div id="chart-nps-prov" style="height:220px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Overall Satisfaction per Layanan<div class="chart-subtitle">Radar skor 1–6</div></div>
          <div id="chart-ovr-radar" style="height:220px"></div>
        </div>
      </div>
      <div class="chart-grid-2">
        <div class="chart-card">
          <div class="chart-title">CSI Distribution<div class="chart-subtitle">Distribusi skor kepuasan real</div></div>
          <div id="chart-csi-dist" style="height:230px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Panel Comparison — Teller vs CS<div class="chart-subtitle">NPS & CSI per panel layanan</div></div>
          <div id="chart-panel-comp" style="height:230px"></div>
        </div>
      </div>
    </div>

    <div class="page" id="page-branch">
      <div class="chart-grid-31">
        <div class="chart-card">
          <div class="chart-title">Branch Ranking — Dot Plot<div class="chart-subtitle">vs. overall average</div></div>
          <div id="chart-branch-dot" style="height:520px"></div>
        </div>
        <div>
          <div class="chart-card" style="margin-bottom:14px">
            <div class="chart-title" style="margin-bottom:10px">Top 5 Branches</div>
            <div class="rank-list" id="rank-top5"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title" style="margin-bottom:10px">Bottom 5 Branches</div>
            <div class="rank-list" id="rank-bot5"></div>
          </div>
        </div>
      </div>
      <div class="chart-grid-2">
        <div class="chart-card">
          <div class="chart-title">NPS per Provinsi<div class="chart-subtitle">Kinerja regional</div></div>
          <div id="chart-prov-bar" style="height:260px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Detail Cabang Terpilih<div class="chart-subtitle">Filter ke 1 cabang untuk detail</div></div>
          <div id="branch-detail" style="padding-top:8px"></div>
        </div>
      </div>
    </div>

    <div class="page" id="page-touchpoint">
      <div class="tab-row" id="tp-tabs"></div>
      <div class="chart-grid-2">
        <div class="chart-card">
          <div class="chart-title">IPA Matrix — <span id="ipa-cat-label">Kantor Cabang</span><div class="chart-subtitle">Importance vs Performance · 4 kuadran strategis</div></div>
          <div id="chart-ipa" style="height:320px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Gap Analysis<div class="chart-subtitle">Performance − Importance per atribut</div></div>
          <div id="chart-gap" style="height:320px"></div>
        </div>
      </div>
      <div class="chart-grid-2">
        <div class="chart-card">
          <div class="chart-title">Radar — Overall Layanan<div class="chart-subtitle">Skor rata-rata per kategori</div></div>
          <div id="chart-tp-radar" style="height:280px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Quick Wins — Prioritas Perbaikan<div class="chart-subtitle">High Importance · Low Performance</div></div>
          <div id="chart-quickwin" style="height:280px"></div>
        </div>
      </div>
    </div>

    <div class="page" id="page-demographics">
      <div class="chart-grid-3">
        <div class="chart-card">
          <div class="chart-title">NPS per Gender</div>
          <div id="chart-gender-nps" style="height:200px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">NPS per Kelompok Usia</div>
          <div id="chart-usia-nps" style="height:200px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Segmentasi Nasabah<div class="chart-subtitle">Loyalitas × frekuensi transaksi</div></div>
          <div id="chart-segmen" style="height:200px"></div>
        </div>
      </div>
      <div class="chart-grid-2">
        <div class="chart-card">
          <div class="chart-title">Emotion Wheel — Bank XYZ<div class="chart-subtitle">% nasabah merasakan kuat (skor ≥5)</div></div>
          <div id="chart-emotion" style="height:300px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">CSI & Loyalty per Demografi<div class="chart-subtitle">Gender comparison</div></div>
          <div id="chart-demo-kpi" style="height:300px"></div>
        </div>
      </div>
    </div>

    <div class="page" id="page-competitor">
      <div class="chart-grid-2">
        <div class="chart-card">
          <div class="chart-title">NPS: Bank XYZ vs Kompetitor<div class="chart-subtitle">Net Promoter Score</div></div>
          <div id="chart-comp-nps" style="height:210px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Overall Satisfaction per Layanan<div class="chart-subtitle">XYZ vs Kompetitor (mean 1–6)</div></div>
          <div id="chart-comp-ovr" style="height:210px"></div>
        </div>
      </div>
      <div class="chart-grid-2">
        <div class="chart-card">
          <div class="chart-title">Brand Perception<div class="chart-subtitle">% setuju atribut brand (skor ≥4)</div></div>
          <div id="chart-brand" style="height:300px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Bank Utama Nasabah<div class="chart-subtitle">Simpan dana vs transaksi</div></div>
          <div id="chart-switch" style="height:300px"></div>
        </div>
      </div>
    </div>

    <div class="page" id="page-recommendation">
      <div class="chart-grid-13">
        <div>
          <div class="chart-card" style="margin-bottom:16px">
            <div class="chart-title">KPI Scorecard</div>
            <div id="scorecard"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">NPS Gauge</div>
            <div id="chart-gauge" style="height:200px"></div>
          </div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Strategic Recommendations<div class="chart-subtitle">Generated from data</div></div>
          <div id="rec-list"></div>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
const D = {data_json};
const C = {{
  orange:'#E85D04',orange2:'#F48C06',orange3:'#FAA307',
  green:'#10B981',red:'#EF4444',blue:'#3B82F6',purple:'#8B5CF6',
  card:'#FFFFFF',bg:'#F5F6FA',border:'#E8ECF4',
  text1:'#1A202C',text2:'#4A5568',text3:'#A0AEC0',
  font:'Inter,sans-serif',fontHead:'Plus Jakarta Sans,sans-serif',
}};
const BL = {{
  paper_bgcolor:C.card,plot_bgcolor:C.card,
  font:{{family:C.font,color:C.text1,size:11}},
  margin:{{t:10,b:30,l:40,r:20}},
  xaxis:{{gridcolor:C.border,zeroline:false,tickfont:{{color:C.text3,size:10}}}},
  yaxis:{{gridcolor:C.border,zeroline:false,tickfont:{{color:C.text3,size:10}}}},
  showlegend:false,
  hoverlabel:{{bgcolor:C.card,bordercolor:C.border,font:{{size:12}}}},
}};
const PC = {{displayModeBar:false,responsive:true}};

let state = {{
  page:'dashboard',
  provs:[...D.all_prov],
  kotas:[...D.all_kota],
  cabangs:[...D.all_cabang],
  panels:[...D.all_panel],
  activeIpaCat: Object.keys(D.ipa_cats)[0] || 'Kantor Cabang',
}};

function toggleSidebar(){{ document.getElementById('sidebar').classList.toggle('collapsed'); }}
function navigate(el){{
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  el.classList.add('active');
  state.page = el.dataset.page;
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.getElementById('page-'+state.page).classList.add('active');
  const T={{dashboard:'Dashboard Overview',branch:'Branch Performance',
    touchpoint:'Touchpoint Analysis',demographics:'Demographics',
    competitor:'Competitor Benchmark',recommendation:'Insight & Recommendation'}};
  document.getElementById('topbar-title').textContent = T[state.page]||state.page;
  buildFilters(state.page);
  renderPage(state.page);
}}

function buildFilters(page){{
  const wrap = document.getElementById('topbar-filters');
  wrap.innerHTML = '';
  if(['dashboard','branch'].includes(page)){{
    wrap.appendChild(buildSelectWidget('f-prov','Provinsi',D.all_prov,state.provs,v=>{{
      state.provs=v;
      state.kotas=[...new Set(v.flatMap(p=>D.prov_kota[p]||[]))];
      state.cabangs=[...new Set(state.kotas.flatMap(k=>D.kota_cabang[k]||[]))];
      buildFilters(page); renderPage(page);
    }}));
    wrap.appendChild(buildSelectWidget('f-kota','Kota/Kab',
      [...new Set(state.provs.flatMap(p=>D.prov_kota[p]||[]))],
      state.kotas, v=>{{
        state.kotas=v;
        state.cabangs=[...new Set(v.flatMap(k=>D.kota_cabang[k]||[]))];
        buildFilters(page); renderPage(page);
    }}));
    wrap.appendChild(buildSelectWidget('f-cabang','Cabang',
      state.cabangs, state.cabangs, v=>{{
        state.cabangs=v; renderPage(page);
    }}));
  }}
  if(['touchpoint','demographics','competitor','recommendation'].includes(page)){{
    wrap.appendChild(buildSelectWidget('f-panel','Panel',D.all_panel,state.panels,v=>{{
      state.panels=v; renderPage(page);
    }}));
  }}
}}

function buildSelectWidget(id,label,opts,selected,onChange){{
  const wrap=document.createElement('div');
  wrap.className='custom-select-wrap';
  const allSel=selected.length===opts.length;
  wrap.innerHTML=`
    <div class="custom-select-btn" id="btn-${{id}}" onclick="toggleDD('${{id}}')">
      <div><div class="select-label">${{label}}</div>
      <div class="select-value" id="val-${{id}}">${{allSel?'All selected':selected.length+' selected'}}</div></div>
      <div class="select-arrow">▼</div>
    </div>
    <div class="custom-dropdown" id="dd-${{id}}">
      <div class="dropdown-header">
        <span class="dropdown-title">${{label}}</span>
        <div class="dropdown-actions">
          <span class="dropdown-action" onclick="ddAll('${{id}}')">All</span>
          <span class="dropdown-action" onclick="ddNone('${{id}}')">None</span>
        </div>
      </div>
      <div class="dropdown-search"><input type="text" placeholder="Search..." oninput="ddFilter('${{id}}',this.value)"></div>
      <div class="dropdown-list" id="list-${{id}}"></div>
    </div>`;
  const listEl=wrap.querySelector('#list-'+id);
  opts.forEach(o=>{{
    const item=document.createElement('div');
    item.className='dropdown-item'+(selected.includes(o)?' checked':'');
    item.dataset.val=o;
    item.dataset.cb=id;
    item.innerHTML=`<div class="custom-checkbox">${{selected.includes(o)?'✓':''}}</div><span>${{o}}</span>`;
    item.onclick=(e)=>{{e.stopPropagation();
      const cur=[...wrap.querySelectorAll('.dropdown-item.checked')].map(i=>i.dataset.val);
      const idx=cur.indexOf(o);
      if(idx>-1){{if(cur.length===1)return;cur.splice(idx,1);item.classList.remove('checked');item.querySelector('.custom-checkbox').textContent='';}}
      else{{cur.push(o);item.classList.add('checked');item.querySelector('.custom-checkbox').textContent='✓';}}
      document.getElementById('val-'+id).textContent=cur.length===opts.length?'All selected':cur.length+' selected';
      onChange(cur);
    }};
    listEl.appendChild(item);
  }});
  return wrap;
}}

function toggleDD(id){{
  const dd=document.getElementById('dd-'+id),btn=document.getElementById('btn-'+id);
  const open=dd.classList.contains('open');
  document.querySelectorAll('.custom-dropdown').forEach(d=>d.classList.remove('open'));
  document.querySelectorAll('.custom-select-btn').forEach(b=>b.classList.remove('open'));
  if(!open){{dd.classList.add('open');btn.classList.add('open');}}
}}
function ddAll(id){{
  document.querySelectorAll('#list-'+id+' .dropdown-item').forEach(i=>{{i.classList.add('checked');i.querySelector('.custom-checkbox').textContent='✓';}});
}}
function ddNone(id){{
  const items=document.querySelectorAll('#list-'+id+' .dropdown-item');
  items.forEach((i,idx)=>{{
    if(idx===0){{i.classList.add('checked');i.querySelector('.custom-checkbox').textContent='✓';}}
    else{{i.classList.remove('checked');i.querySelector('.custom-checkbox').textContent='';}}
  }});
}}
function ddFilter(id,q){{
  document.querySelectorAll('#list-'+id+' .dropdown-item').forEach(i=>{{
    i.style.display=i.dataset.val.toLowerCase().includes(q.toLowerCase())?'':'none';
  }});
}}
document.addEventListener('click',()=>{{
  document.querySelectorAll('.custom-dropdown').forEach(d=>d.classList.remove('open'));
  document.querySelectorAll('.custom-select-btn').forEach(b=>b.classList.remove('open'));
}});

function npsColor(v){{return v>=50?C.green:v>=20?C.orange:v>=0?C.orange2:C.red;}}
function filtBranch(){{return D.branch.filter(b=>state.cabangs.includes(b.CABANG));}}
function filtProv(){{return D.prov.filter(p=>state.provs.includes(p.PROV));}}

function renderPage(p){{
  if(p==='dashboard')    renderDashboard();
  if(p==='branch')       renderBranch();
  if(p==='touchpoint')   renderTouchpoint();
  if(p==='demographics') renderDemographics();
  if(p==='competitor')   renderCompetitor();
  if(p==='recommendation') renderRecommendation();
}}

function renderDashboard(){{
  const g=D.global_kpi;
  const kpis=[
    {{l:'NPS SCORE',v:g.nps.toFixed(1),b:g.nps>=50?'Excellent':g.nps>=20?'Good':'Needs Work',
      c:g.nps>=50?'badge-green':g.nps>=20?'badge-orange':'badge-red',col:npsColor(g.nps)}},
    {{l:'CSI MEAN',v:g.csi+'/6',b:g.csi_pct_puas+'% Sangat Puas',
      c:g.csi>=5.5?'badge-green':g.csi>=5.0?'badge-orange':'badge-red',col:C.green}},
    {{l:'LOYALTY INDEX',v:g.loyalty+'/6',b:'Rata-rata semua panel',c:'badge-blue',col:C.blue}},
    {{l:'TOTAL RESPONDEN',v:g.total.toLocaleString(),b:g.provinces+' provinsi · '+g.branches+' cabang',c:'badge-orange',col:C.orange}},
    {{l:'PROMOTER vs DETRACTOR',v:((g.promoters/g.total)*100).toFixed(1)+'%',
      b:'Det: '+((g.detractors/g.total)*100).toFixed(1)+'%',c:'badge-purple',col:C.purple}},
  ];
  document.getElementById('kpi-grid').innerHTML=kpis.map((k,i)=>`
    <div class="kpi-card" style="--c:${{k.col}}">
      <style>#kpi-grid .kpi-card:nth-child(${{i+1}})::before{{background:${{k.col}}}}</style>
      <div class="kpi-card-label">${{k.l}}</div>
      <div class="kpi-card-value" style="color:${{k.col}}">${{k.v}}</div>
      <span class="kpi-badge ${{k.c}}">${{k.b}}</span>
    </div>`).join('');

  const prom=g.promoters,pass=g.passives,det=g.detractors;
  Plotly.newPlot('chart-nps-donut',[{{
    type:'pie',labels:['Promoter','Passive','Detractor'],values:[prom,pass,det],hole:.58,
    marker:{{colors:[C.green,'#94A3B8',C.red]}},
    textinfo:'percent+label',textfont:{{size:11}},
    hovertemplate:'<b>%{{label}}</b><br>Count: %{{value}}<br>%{{percent}}<extra></extra>',
  }}],{{...BL,margin:{{t:10,b:10,l:10,r:10}},showlegend:true,
    legend:{{orientation:'v',x:.82,y:.5,font:{{size:10}}}},
    annotations:[{{text:`<b>${{g.nps}}</b><br>NPS`,x:.5,y:.5,showarrow:false,font:{{size:14,color:npsColor(g.nps)}}}}],
  }},PC);

  const pv=filtProv().sort((a,b)=>a.nps_score-b.nps_score);
  Plotly.newPlot('chart-nps-prov',[{{
    type:'bar',orientation:'h',x:pv.map(p=>p.nps_score),y:pv.map(p=>p.PROV),
    marker:{{color:pv.map(p=>npsColor(p.nps_score))}},
    text:pv.map(p=>p.nps_score.toFixed(1)),textposition:'outside',
    hovertemplate:'%{{y}}: NPS %{{x}}<extra></extra>',
  }}],{{...BL,margin:{{t:10,b:30,l:130,r:50}},
    xaxis:{{...BL.xaxis,range:[Math.min(...pv.map(p=>p.nps_score))-20,130]}},
    shapes:[{{type:'line',x0:g.nps,x1:g.nps,y0:-.5,y1:pv.length-.5,
              line:{{color:C.orange,dash:'dot',width:1.5}}}}],
  }},PC);

  const ov=D.ovr_radar;
  const labels=ov.map(o=>o.kategori_layanan),vals=ov.map(o=>o.mean_score);
  Plotly.newPlot('chart-ovr-radar',[{{
    type:'scatterpolar',mode:'lines+markers',
    r:[...vals,vals[0]],theta:[...labels,labels[0]],
    fill:'toself',fillcolor:'rgba(232,93,4,.15)',
    line:{{color:C.orange,width:2}},marker:{{color:C.orange,size:6}},
    hovertemplate:'%{{theta}}: %{{r:.2f}}<extra></extra>',
  }}],{{paper_bgcolor:C.card,plot_bgcolor:C.card,
    polar:{{radialaxis:{{visible:true,range:[4.5,6.2],tickfont:{{size:9}}}},
            angularaxis:{{tickfont:{{size:9}}}}}},
    margin:{{t:20,b:20,l:30,r:30}},font:{{family:C.font}},showlegend:false,
  }},PC);

  const csi=D.csi_dist;
  Plotly.newPlot('chart-csi-dist',[{{
    type:'bar',x:csi.scores,y:csi.counts,
    marker:{{color:csi.scores.map(s=>s<=2?C.red:s===3?'#94A3B8':s===4?C.orange:s===5?C.orange2:C.green),
             line:{{color:'white',width:1}}}},
    text:csi.counts.map(v=>((v/g.total)*100).toFixed(1)+'%'),textposition:'outside',textfont:{{size:10}},
    hovertemplate:'Skor %{{x}}: %{{y}} responden<extra></extra>',
  }}],{{...BL,margin:{{t:20,b:30,l:40,r:10}},
    xaxis:{{...BL.xaxis,title:{{text:'CSI Score',font:{{size:10}}}},tickvals:csi.scores}},
    yaxis:{{...BL.yaxis,title:{{text:'Count',font:{{size:10}}}}}},
  }},PC);

  const pk=D.panel_kpi, pk_keys=Object.keys(pk);
  Plotly.newPlot('chart-panel-comp',[
    {{type:'bar',name:'NPS Score',x:pk_keys,y:pk_keys.map(k=>pk[k].nps),
      marker:{{color:[C.orange,C.blue]}},text:pk_keys.map(k=>pk[k].nps.toFixed(1)),textposition:'outside',yaxis:'y'}},
    {{type:'scatter',mode:'markers+lines',name:'CSI Mean',x:pk_keys,y:pk_keys.map(k=>pk[k].csi),
      yaxis:'y2',line:{{color:C.green,width:2}},marker:{{size:10,color:C.green}}}},
  ],{{...BL,showlegend:true,
    legend:{{orientation:'h',y:1.15,font:{{size:10}}}},
    margin:{{t:30,b:40,l:50,r:60}},
    yaxis:{{...BL.yaxis,title:'NPS Score',range:[0,120]}},
    yaxis2:{{overlaying:'y',side:'right',range:[5,6.2],title:'CSI Mean',tickfont:{{size:10}}}},
  }},PC);
}}

function renderBranch(){{
  const br=filtBranch().sort((a,b)=>b.nps_score-a.nps_score);
  const avgNPS=D.global_kpi.nps;

  const traces=[];
  br.forEach(b=>{{
    traces.push({{type:'scatter',mode:'markers+text',
      x:[b.nps_score],y:[b.CABANG],
      marker:{{size:14,color:b.nps_score>=avgNPS?C.green:C.red,line:{{color:'white',width:2}}}},
      text:[b.nps_score.toFixed(1)],textposition:'middle right',textfont:{{size:9,color:C.text2}},
      hovertemplate:`<b>${{b.CABANG}}</b><br>NPS: ${{b.nps_score}}<br>CSI: ${{b.csi_mean?.toFixed(2)}}<br>Responden: ${{b.n_responden}}<extra></extra>`,
      showlegend:false}});
    traces.push({{type:'scatter',mode:'lines',x:[0,b.nps_score],y:[b.CABANG,b.CABANG],
      line:{{color:'rgba(200,200,200,.4)',width:1}},showlegend:false,hoverinfo:'skip'}});
  }});
  Plotly.newPlot('chart-branch-dot',traces,{{...BL,showlegend:false,
    height:Math.max(460,br.length*22),
    margin:{{t:20,b:20,l:150,r:60}},
    xaxis:{{...BL.xaxis,title:{{text:'NPS Score',font:{{size:10}}}},showgrid:false}},
    yaxis:{{...BL.yaxis,autorange:'reversed',tickfont:{{size:9,color:C.text2}}}},
    shapes:[{{type:'line',x0:avgNPS,x1:avgNPS,y0:-.5,y1:br.length-.5,
              line:{{color:C.orange,width:1.5,dash:'dot'}}}}],
    annotations:[{{x:avgNPS,y:-.5,text:`Avg: ${{avgNPS}}`,showarrow:false,
                   font:{{size:9,color:C.orange}},yanchor:'top'}}],
  }},PC);

  const rc=['r1','r2','r3','rn','rn'];
  const top5=br.slice(0,5), bot5=[...br].sort((a,b)=>a.nps_score-b.nps_score).slice(0,5);
  document.getElementById('rank-top5').innerHTML=top5.map((b,i)=>`
    <div class="rank-item">
      <div class="rank-num ${{rc[i]}}">#${{i+1}}</div>
      <div style="flex:1;min-width:0"><div class="rank-name">${{b.CABANG}}</div>
        <div class="rank-sub">${{b.n_responden}} responden · ${{b.PROV}}</div></div>
      <div class="rank-score">${{b.nps_score.toFixed(1)}}</div>
    </div>`).join('');
  document.getElementById('rank-bot5').innerHTML=bot5.map((b,i)=>`
    <div class="rank-item" style="border-left:3px solid ${{C.red}}">
      <div class="rank-num rn" style="background:rgba(239,68,68,.1);color:${{C.red}}">↓${{br.length-i}}</div>
      <div style="flex:1;min-width:0"><div class="rank-name">${{b.CABANG}}</div>
        <div class="rank-sub">${{b.n_responden}} responden · ${{b.PROV}}</div></div>
      <div class="rank-score" style="color:${{C.red}}">${{b.nps_score.toFixed(1)}}</div>
    </div>`).join('');

  const pv=filtProv().sort((a,b)=>b.nps_score-a.nps_score);
  Plotly.newPlot('chart-prov-bar',[{{
    type:'bar',x:pv.map(p=>p.PROV),y:pv.map(p=>p.nps_score),
    marker:{{color:pv.map(p=>npsColor(p.nps_score))}},
    text:pv.map(p=>p.nps_score.toFixed(1)),textposition:'outside',
    hovertemplate:'%{{x}}: NPS %{{y}}<extra></extra>',
  }}],{{...BL,margin:{{t:30,b:80,l:50,r:20}},
    yaxis:{{...BL.yaxis,range:[Math.max(0,Math.min(...pv.map(p=>p.nps_score))-20),130]}},
    xaxis:{{...BL.xaxis,tickangle:-30}},
    shapes:[{{type:'line',x0:-.5,x1:pv.length-.5,y0:D.global_kpi.nps,y1:D.global_kpi.nps,
              line:{{color:C.orange,dash:'dot',width:1.5}}}}],
  }},PC);

  const det=document.getElementById('branch-detail');
  if(state.cabangs.length===1){{
    const b=br.find(x=>x.CABANG===state.cabangs[0]);
    if(b) det.innerHTML=`
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        ${{[['NPS Score',b.nps_score.toFixed(1),npsColor(b.nps_score),'Rank #'+b.rank+' dari '+br.length],
           ['CSI Mean',(b.csi_mean||0).toFixed(2),C.green,'Rata-rata layanan'],
           ['Loyalty',(b.loyalty_mean||0).toFixed(2),C.blue,'Index loyalitas'],
           ['Responden',b.n_responden,C.orange,b.PROV+' · '+b.KABKOTA]
         ].map(([l,v,c,s])=>`<div style="background:var(--bg);border-radius:10px;padding:10px 12px;border:1px solid var(--border)">
           <div style="font-size:10px;color:var(--text-3)">${{l}}</div>
           <div style="font-size:20px;font-weight:700;color:${{c}}">${{v}}</div>
           <div style="font-size:9px;color:var(--text-3)">${{s}}</div>
         </div>`).join('')}}
      </div>`;
  }} else {{
    det.innerHTML=`<div style="padding:30px;text-align:center;color:var(--text-3);font-size:12px">
      Pilih <b>1 cabang</b> dari filter untuk melihat detail metrik</div>`;
  }}
}}

function renderTouchpoint(){{
  const cats=Object.keys(D.ipa_cats);
  document.getElementById('tp-tabs').innerHTML=cats.map(c=>`
    <div class="tab-btn${{c===state.activeIpaCat?' active':''}}" onclick="setIpaCat('${{c}}')">${{c}}</div>`).join('');
  renderIPA(state.activeIpaCat);

  const ov=D.ovr_radar;
  const lb=ov.map(o=>o.kategori_layanan),vl=ov.map(o=>o.mean_score);
  Plotly.newPlot('chart-tp-radar',[{{
    type:'scatterpolar',mode:'lines+markers',
    r:[...vl,vl[0]],theta:[...lb,lb[0]],
    fill:'toself',fillcolor:'rgba(232,93,4,.15)',
    line:{{color:C.orange,width:2}},marker:{{color:C.orange,size:7}},
    hovertemplate:'%{{theta}}: %{{r:.2f}}<extra></extra>',
  }}],{{paper_bgcolor:C.card,
    polar:{{radialaxis:{{visible:true,range:[4.5,6.2],tickfont:{{size:9}}}},
            angularaxis:{{tickfont:{{size:9}}}}}},
    margin:{{t:20,b:20,l:30,r:30}},font:{{family:C.font}},showlegend:false,
  }},PC);

  const qw=Object.values(D.ipa_cats).flat().filter(a=>a.kuadran==='Quick Win')
            .sort((a,b)=>b.importance-a.importance).slice(0,10);
  if(qw.length>0){{
    Plotly.newPlot('chart-quickwin',[{{
      type:'bar',orientation:'h',
      x:qw.map(a=>a.importance),
      y:qw.map(a=>(a.atribut||'').substring(0,40)),
      marker:{{color:C.red,opacity:.85}},
      text:qw.map(a=>a.importance.toFixed(2)),textposition:'outside',
      hovertemplate:'%{{y}}<br>Importance: %{{x:.2f}}<extra></extra>',
    }}],{{...BL,margin:{{t:10,b:30,l:250,r:60}},
      xaxis:{{...BL.xaxis,title:'Importance Score'}},
      yaxis:{{...BL.yaxis,autorange:'reversed'}},
    }},PC);
  }}
}}

function setIpaCat(cat){{
  state.activeIpaCat=cat;
  document.querySelectorAll('#tp-tabs .tab-btn').forEach(t=>t.classList.toggle('active',t.textContent===cat));
  document.getElementById('ipa-cat-label').textContent=cat;
  renderIPA(cat);
}}

function renderIPA(cat){{
  const items=D.ipa_cats[cat]||[];
  if(!items.length) return;
  const qC={{'Keep Up':C.green,'Quick Win':C.red,'Low Priority':'#94A3B8','Possible Overkill':C.blue}};
  const impMed=items.reduce((s,a)=>s+a.importance,0)/items.length;
  const satMed=items.reduce((s,a)=>s+a.performance,0)/items.length;

  Plotly.newPlot('chart-ipa',[{{
    type:'scatter',mode:'markers',
    x:items.map(a=>a.performance),y:items.map(a=>a.importance),
    marker:{{color:items.map(a=>qC[a.kuadran]),size:10,opacity:.85,line:{{color:'white',width:1}}}},
    text:items.map(a=>(a.atribut||'').substring(0,20)),
    hovertemplate:'<b>%{{text}}</b><br>Importance: %{{y:.2f}}<br>Performance: %{{x:.2f}}<extra></extra>',
  }}],{{...BL,margin:{{t:30,b:40,l:60,r:20}},
    xaxis:{{...BL.xaxis,title:'Performance (Satisfaction)'}},
    yaxis:{{...BL.yaxis,title:'Importance'}},
    shapes:[
      {{type:'line',x0:satMed,x1:satMed,y0:Math.min(...items.map(a=>a.importance))-.3,y1:Math.max(...items.map(a=>a.importance))+.3,line:{{color:C.border,dash:'dash',width:1}}}},
      {{type:'line',x0:Math.min(...items.map(a=>a.performance))-.3,x1:Math.max(...items.map(a=>a.performance))+.3,y0:impMed,y1:impMed,line:{{color:C.border,dash:'dash',width:1}}}},
    ],
    annotations:[
      {{x:Math.min(...items.map(a=>a.performance)),y:Math.max(...items.map(a=>a.importance)),text:'<b>Quick Win</b>',showarrow:false,font:{{size:9,color:C.red}},xanchor:'left'}},
      {{x:Math.max(...items.map(a=>a.performance)),y:Math.max(...items.map(a=>a.importance)),text:'<b>Keep Up</b>',showarrow:false,font:{{size:9,color:C.green}},xanchor:'right'}},
      {{x:Math.min(...items.map(a=>a.performance)),y:Math.min(...items.map(a=>a.importance)),text:'<b>Low Priority</b>',showarrow:false,font:{{size:9,color:'#94A3B8'}},xanchor:'left'}},
      {{x:Math.max(...items.map(a=>a.performance)),y:Math.min(...items.map(a=>a.importance)),text:'<b>Possible Overkill</b>',showarrow:false,font:{{size:9,color:C.blue}},xanchor:'right'}},
    ],
  }},PC);

  const sorted=[...items].sort((a,b)=>a.gap-b.gap);
  Plotly.newPlot('chart-gap',[{{
    type:'bar',orientation:'h',
    x:sorted.map(a=>a.gap),y:sorted.map(a=>(a.atribut||'').substring(0,35)),
    marker:{{color:sorted.map(a=>a.gap>=0?C.green:C.red),opacity:.85}},
    text:sorted.map(a=>a.gap.toFixed(3)),textposition:'outside',
    hovertemplate:'%{{y}}<br>Gap: %{{x:.3f}}<extra></extra>',
  }}],{{...BL,
    height:Math.max(280,sorted.length*18),
    margin:{{t:10,b:40,l:220,r:70}},
    xaxis:{{...BL.xaxis,title:'Performance − Importance',zeroline:true,zerolinecolor:C.border}},
    yaxis:{{...BL.yaxis,autorange:'reversed'}},
  }},PC);
}}

function renderDemographics(){{
  Plotly.newPlot('chart-gender-nps',[{{
    type:'bar',x:D.gender.map(g=>g.gender),y:D.gender.map(g=>g.nps_score),
    marker:{{color:[C.blue,'#DB2777']}},
    text:D.gender.map(g=>g.nps_score.toFixed(1)),textposition:'outside',textfont:{{size:10,color:C.text2}},
    hovertemplate:'<b>%{{x}}</b><br>NPS: %{{y:.1f}}<extra></extra>',
  }}],{{...BL,margin:{{t:10,b:30,l:40,r:10}},
    yaxis:{{...BL.yaxis,range:[0,120]}},
  }},PC);

  Plotly.newPlot('chart-usia-nps',[{{
    type:'bar',x:D.usia.map(u=>u.usia_group),y:D.usia.map(u=>u.nps_score),
    marker:{{color:D.usia.map(u=>npsColor(u.nps_score))}},
    text:D.usia.map(u=>u.nps_score.toFixed(1)),textposition:'outside',textfont:{{size:10,color:C.text2}},
    hovertemplate:'<b>%{{x}}</b><br>NPS: %{{y:.1f}}<extra></extra>',
  }}],{{...BL,margin:{{t:10,b:50,l:40,r:10}},
    yaxis:{{...BL.yaxis,range:[0,120]}},
    xaxis:{{...BL.xaxis,tickangle:-30}},
  }},PC);

  Plotly.newPlot('chart-segmen',[{{
    type:'pie',labels:D.segmen.map(s=>s.segmen),values:D.segmen.map(s=>s.n),
    hole:.45,marker:{{colors:[C.green,C.orange,C.blue,C.red,'#94A3B8']}},
    textinfo:'label+percent',textfont:{{size:10}},
    hovertemplate:'%{{label}}: %{{value}} (%{{percent}})<extra></extra>',
  }}],{{...BL,margin:{{t:10,b:10,l:10,r:10}}}},PC);

  const ae=[...D.emo_pos,...D.emo_neg];
  Plotly.newPlot('chart-emotion',[{{
    type:'scatterpolar',mode:'lines+markers',
    r:[...ae.map(e=>e.pct_strong),ae[0]?.pct_strong||0],
    theta:[...ae.map(e=>e.emosi),ae[0]?.emosi||''],
    fill:'toself',fillcolor:'rgba(232,93,4,.12)',
    line:{{color:C.orange,width:2}},
    marker:{{color:ae.map(e=>D.emo_pos.includes(e)?C.green:C.red).concat([C.green]),size:6}},
    hovertemplate:'<b>%{{theta}}</b><br>%{{r:.1f}}% merasakan kuat<extra></extra>',
  }}],{{paper_bgcolor:C.card,
    polar:{{radialaxis:{{visible:true,range:[0,100],tickfont:{{size:8}}}},angularaxis:{{tickfont:{{size:9}}}}}},
    margin:{{t:20,b:20,l:20,r:20}},font:{{family:C.font}},showlegend:false,
  }},PC);

  Plotly.newPlot('chart-demo-kpi',[
    {{type:'bar',name:'CSI',x:D.gender.map(g=>g.gender),y:D.gender.map(g=>g.csi_mean),marker:{{color:C.orange,opacity:.85}}}},
    {{type:'bar',name:'Loyalty',x:D.gender.map(g=>g.gender),y:D.gender.map(g=>g.loyalty_mean),marker:{{color:C.green,opacity:.85}}}},
  ],{{...BL,barmode:'group',showlegend:true,
    legend:{{orientation:'h',y:1.1,font:{{size:10}}}},
    margin:{{t:40,b:40,l:50,r:20}},
    yaxis:{{...BL.yaxis,range:[5,6.5],title:{{text:'Score (1–6)',font:{{size:10}}}}}},
  }},PC);
}}

function renderCompetitor(){{
  Plotly.newPlot('chart-comp-nps',[{{
    type:'bar',x:D.nps_comp.map(b=>b.bank),y:D.nps_comp.map(b=>b.nps_score||0),
    marker:{{color:[C.orange,C.blue]}},
    text:D.nps_comp.map(b=>(b.nps_score||0).toFixed(1)),textposition:'outside',textfont:{{size:10}},
    hovertemplate:'<b>%{{x}}</b><br>NPS: %{{y:.1f}}<extra></extra>',
  }}],{{...BL,margin:{{t:30,b:40,l:50,r:20}},
    yaxis:{{...BL.yaxis,range:[0,120]}},
  }},PC);

  Plotly.newPlot('chart-comp-ovr',[
    {{type:'bar',name:'Bank XYZ',x:D.comp.map(c=>c.kategori),y:D.comp.map(c=>c.xyz_mean||0),marker:{{color:C.orange,opacity:.85}}}},
    {{type:'bar',name:'Kompetitor',x:D.comp.map(c=>c.kategori),y:D.comp.map(c=>c.komp_mean||0),marker:{{color:C.blue,opacity:.7}}}},
  ],{{...BL,barmode:'group',showlegend:true,
    legend:{{orientation:'h',y:1.1,font:{{size:10}}}},
    margin:{{t:40,b:80,l:50,r:20}},
    yaxis:{{...BL.yaxis,range:[4.5,6.5]}},
    xaxis:{{...BL.xaxis,tickangle:-25}},
  }},PC);

  const bd=D.brand.sort((a,b)=>(b.xyz_pct_agree||b.xyz_pct||0)-(a.xyz_pct_agree||a.xyz_pct||0));
  Plotly.newPlot('chart-brand',[
    {{type:'bar',name:'Bank XYZ',x:bd.map(b=>b.atribut.substring(0,38)),y:bd.map(b=>b.xyz_pct_agree||b.xyz_pct||0),marker:{{color:C.orange,opacity:.85}}}},
    {{type:'bar',name:'Kompetitor',x:bd.map(b=>b.atribut.substring(0,38)),y:bd.map(b=>b.komp_pct_agree||b.komp_pct||0),marker:{{color:C.blue,opacity:.7}}}},
  ],{{...BL,barmode:'group',showlegend:true,
    legend:{{font:{{size:10}}}},
    margin:{{t:30,b:130,l:50,r:20}},
    xaxis:{{...BL.xaxis,tickangle:-35,tickfont:{{size:9}}}},
    yaxis:{{...BL.yaxis,range:[0,110],title:{{text:'% Setuju',font:{{size:10}}}}}},
  }},PC);

  const simp=D.switch.filter(s=>s.tipe==='Bank Simpan Utama').slice(0,5);
  const tran=D.switch.filter(s=>s.tipe==='Bank Transaksi Utama').slice(0,5);
  Plotly.newPlot('chart-switch',[
    {{type:'bar',name:'Bank Simpan',x:simp.map(s=>s.bank.replace('Bank ','').substring(0,14)),y:simp.map(s=>s.pct),marker:{{color:C.orange,opacity:.85}}}},
    {{type:'bar',name:'Bank Transaksi',x:tran.map(s=>s.bank.replace('Bank ','').substring(0,14)),y:tran.map(s=>s.pct),marker:{{color:C.blue,opacity:.7}}}},
  ],{{...BL,barmode:'group',showlegend:true,
    legend:{{font:{{size:10}}}},
    margin:{{t:30,b:80,l:50,r:20}},
    yaxis:{{...BL.yaxis,title:{{text:'%',font:{{size:10}}}},range:[0,100]}},
    xaxis:{{...BL.xaxis,tickangle:-20}},
  }},PC);
}}

function renderRecommendation(){{
  const g=D.global_kpi;
  const loyMean=D.gender.reduce((a,b)=>a+b.loyalty_mean*b.n,0)/D.gender.reduce((a,b)=>a+b.n,0);

  const scores=[
    {{m:'NPS Score',v:g.nps.toFixed(1),t:'Target: 50',ok:g.nps>=50,near:g.nps>=30,color:''}},
    {{m:'CSI Mean',v:g.csi+'/6',t:'Target: 5.5',ok:g.csi>=5.5,near:g.csi>=5.0,color:''}},
    {{m:'Loyalty Index',v:loyMean.toFixed(2)+'/6',t:'Target: 5.5',ok:loyMean>=5.5,near:loyMean>=5.0,color:''}},
    {{m:'Total Responden',v:g.total.toLocaleString(),t:g.branches+' cabang',ok:true,near:true,color:''}},
  ];
  document.getElementById('scorecard').innerHTML=scores.map(s=>{{
    const color=s.ok?C.green:s.near?C.orange:C.red;
    const status=s.ok?'On Target':s.near?'Near Target':'Below Target';
    return `<div class="score-item">
      <div class="score-left"><div class="score-metric">${{s.m}}</div><div class="score-target">${{s.t}}</div></div>
      <div class="score-right"><div class="score-val" style="color:${{color}}">${{s.v}}</div>
      <div class="score-status" style="color:${{color}}">${{status}}</div></div>
    </div>`;
  }}).join('');

  Plotly.newPlot('chart-gauge',[{{
    type:'indicator',mode:'gauge+number+delta',value:g.nps,
    delta:{{reference:50,increasing:{{color:C.green}},decreasing:{{color:C.red}}}},
    gauge:{{
      axis:{{range:[-100,100],tickwidth:1,tickcolor:C.text3,tickfont:{{size:9}}}},
      bar:{{color:npsColor(g.nps),thickness:.25}},bgcolor:'#F5F6FA',borderwidth:0,
      steps:[
        {{range:[-100,0],color:'rgba(239,68,68,.1)'}},
        {{range:[0,30],color:'rgba(245,158,11,.1)'}},
        {{range:[30,70],color:'rgba(16,185,129,.1)'}},
        {{range:[70,100],color:'rgba(16,185,129,.2)'}},
      ],
      threshold:{{line:{{color:C.green,width:3}},thickness:.75,value:50}},
    }},
    number:{{font:{{size:28,family:C.fontHead,color:C.text1}}}},
    title:{{text:'Current NPS Score',font:{{size:11,color:C.text3}}}},
  }}],{{paper_bgcolor:C.card,height:200,margin:{{t:20,b:10,l:20,r:20}},font:{{family:C.font}}}},PC);

  const qw=Object.values(D.ipa_cats).flat().filter(a=>a.kuadran==='Quick Win').sort((a,b)=>b.importance-a.importance);
  const bot=D.branch.slice(-3).reverse();
  const top=D.branch[0];
  const recs=[
    ['PRIORITY 1','Perbaiki Quick Win Touchpoint',
     `<b>${{(qw[0]?.atribut||'').substring(0,50)}}</b> memiliki importance tinggi namun performance rendah. Investasi training dan redesain proses di area ini memberikan dampak terbesar pada NPS.`],
    ['PRIORITY 2','Detractor Recovery Program',
     `<b>${{g.detractors.toLocaleString()}}</b> Detractor (${{(g.detractors/g.total*100).toFixed(1)}}%) merupakan risiko churn segera. Implementasi program follow-up proaktif dengan outreach personal dan service recovery.`],
    ['PRIORITY 3','Intervensi Cabang Kinerja Rendah',
     `Cabang <b>${{bot.map(b=>b.CABANG).join(', ')}}</b> menunjukkan NPS terendah. Lakukan benchmark terhadap best practice cabang top performer.`],
    ['PRIORITY 4','Replikasi Best Practice',
     `Cabang <b>${{top?.CABANG}}</b> (NPS ${{top?.nps_score?.toFixed(1)}}) memimpin semua cabang. Dokumentasikan SOP layanan dan replikasi ke cabang dengan kinerja di bawah rata-rata.`],
    ['PRIORITY 5','Leverage Promoter sebagai Brand Ambassador',
     `<b>${{g.promoters.toLocaleString()}}</b> Promoter (${{(g.promoters/g.total*100).toFixed(1)}}%) adalah aset terbesar. Program referral & loyalty reward dapat memperkuat word-of-mouth secara signifikan.`],
  ];
  document.getElementById('rec-list').innerHTML=recs.map(([n,t,b])=>`
    <div class="rec-card">
      <div class="rec-num">${{n}}</div>
      <div class="rec-title">${{t}}</div>
      <div class="rec-body">${{b}}</div>
    </div>`).join('');
}}

buildFilters('dashboard');
renderPage('dashboard');
</script>
</body></html>"""


# ==========================================
# 5. AI LOGIC LAYER (GROQ)
# ==========================================
def generate_ai_context(data: Dict[str, Any]) -> str:
    """Mengekstraksi data ke dalam narasi string agar Groq memiliki konteks data operasional."""
    g_kpi = data['global_kpi']

    # Kumpulkan Quick Wins Prioritas Teratas
    qw_list = [item for sublist in data['ipa_cats'].values() for item in sublist if item['kuadran'] == 'Quick Win']
    qw_sorted = sorted(qw_list, key=lambda x: x['importance'], reverse=True)[:3]
    
    # Kinerja Cabang
    top_branch = data['branch'][0]['CABANG'] if data['branch'] else 'N/A'
    bot_branch = data['branch'][-1]['CABANG'] if data['branch'] else 'N/A'
    
    # Sinyal Emosi
    top_emo = ", ".join([e['emosi'] for e in data['emo_pos'][:3]])

    # Ekstraksi Analisis Driver (Korelasi) Teratas
    top_drivers_list = sorted(data['driver_analysis'], key=lambda x: abs(x.get('correlation', 0)), reverse=True)[:3]
    driver_str = "; ".join([f"{d['touchpoint']} (r={d['correlation']})" for d in top_drivers_list])

    # Rangkuman Waktu Tunggu Antrean jika ada
    wait_str = "Data antrean stabil"
    if data['wait_time']:
        top_wait = sorted(data['wait_time'], key=lambda x: x.get('count', 0), reverse=True)[:2]
        wait_str = ", ".join([f"{w['panel']} {w['wait_category']} ({w['pct']}%)" for w in top_wait])

    return f"""Konteks data operasional hasil analitik super lengkap untuk BANK XYZ:
- Responden Valid: {g_kpi['total']} nasabah, mencakup {g_kpi['provinces']} provinsi & {g_kpi['branches']} kantor cabang.
- KPI Finansial & Loyalitas: NPS Score global {g_kpi['nps']:.1f} (Promoter: {g_kpi['promoters']}, Detractor: {g_kpi['detractors']}).
- Kepuasan Global: CSI Mean {g_kpi['csi']}/6 ({g_kpi['csi_pct_puas']}% Sangat Puas) | Loyalty Index: {g_kpi['loyalty']}/6.
- Korelasi Driver Utama (Touchpoint vs NPS): {driver_str}.
- Analisis Waktu Tunggu Terlama: {wait_str}.
- Pemetaan Regional: Kantor cabang performa nomor 1 dipimpin oleh {top_branch}, sedangkan performa kritis terbawah berada di {bot_branch}.
- Atribut Prioritas Utama Perbaikan (Quick Wins): {"; ".join([q['atribut'][:40] for q in qw_sorted]) if qw_sorted else 'Tidak ada atribut kritis'}.
- Sinyal Emosi Utama Nasabah: {top_emo}.
- Komparasi Market Share: Skor NPS Bank XYZ {data['nps_comp'][0]['nps_score'] if data['nps_comp'] else 'N/A'} vs Kompetitor Pasar {data['nps_comp'][1]['nps_score'] if len(data['nps_comp']) > 1 else 'N/A'}.
"""

def render_ai_analyst_interface(data_context: str) -> None:
    """Me-render UI Streamlit Chatbot yang aman menggunakan Groq."""
    st.write("---")
    st.subheader("🧠 AI Analyst 🤖 (Secure Backend powered by Groq)")

    # Konfigurasi API
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        groq_key = st.text_input("🔑 Groq API Key Anda belum dikonfigurasi di Env. Masukkan kunci Anda di sini:", type="password", placeholder="gsk_...")

    if not groq_key:
        st.info("💡 Sediakan kunci GROQ_API_KEY Anda untuk mengaktifkan modul tanya jawab AI Analyst secara aman.")
        return

    # Inisialisasi Client
    client = Groq(api_key=groq_key)
    
    # State Management untuk riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo Shafi! Saya sudah membaca seluruh data analitik dari pengujian notebook Anda. Bagian mana dari data Bank XYZ yang ingin Anda kita bedah secara mendalam?"}
        ]

    st.write("**Pertanyaan Populer:**")
    chips = [
        'Apa insight terpenting dari data ini?',
        'Touchpoint apa yang paling berpengaruh ke NPS?',
        'Cabang mana yang perlu prioritas perbaikan?',
        'Buatkan ringkasan eksekutif untuk direksi'
    ]
    
    cols_chips = st.columns(len(chips))
    chosen_prompt = None
    for idx, prompt_text in enumerate(chips):
        if cols_chips[idx].button(prompt_text, key=f"chip_{idx}"):
            chosen_prompt = prompt_text

    # Menampilkan log percakapan
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Input User
    user_input = st.chat_input("Tanya tentang data, cabang, touchpoint, atau rekomendasi strategi...")
    if chosen_prompt:
        user_input = chosen_prompt

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("AI Analyst sedang mengevaluasi data survei..."):
                system_prompt = """Kamu adalah AI Analyst profesional untuk dashboard kepuasan nasabah Bank XYZ.
Jawab seluruh pertanyaan dalam Bahasa Indonesia yang formal, ringkas, solutif, dan wajib menyertakan data angka spesifik dan hasil nilai korelasi (r) atau persentase dari konteks data analitik yang disediakan.
Jangan pernah menyebut nama bank asli, selalu sebut 'Bank XYZ'. Maksimal 3 paragraf jawaban."""
                
                try:
                    chat_completion = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt + f"\n\nContext:\n{data_context}"},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=0.3,
                        max_tokens=800
                    )
                    response_text = chat_completion.choices[0].message.content
                    st.write(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as err:
                    st.error(f"Gagal memanggil Groq API: {str(err)}")


# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main() -> None:
    """Fungsi utama yang merangkai seluruh alur eksekusi aplikasi."""
    # 1. Setup Halaman
    setup_streamlit_page()
    
    # 2. Load Data Analitik Lengkap
    data = load_analytical_data()
    data_json = json.dumps(data, ensure_ascii=False, default=str)
    
    # 3. Merender Desain Dashboard HTML Interaktif Kustom (Tanpa Merusak Desain Asli)
    html_dashboard = get_dashboard_html(data_json)
    components.html(html_dashboard, height=640, scrolling=True) 
    
    # 4. Merender Bot AI yang Dilengkapi Konteks Data Penuh
    data_context = generate_ai_context(data)
    render_ai_analyst_interface(data_context)

if __name__ == "__main__":
    main()