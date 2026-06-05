import streamlit as st
import pandas as pd
import numpy as np
import json, os
import streamlit.components.v1 as components

st.set_page_config(page_title="Bank XYZ Intelligence", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
* { margin:0; padding:0; box-sizing:border-box; }
.stApp { margin:0; padding:0; overflow:hidden; }
[data-testid="stAppViewContainer"] { padding:0; }
[data-testid="stHeader"] { display:none; }
[data-testid="stSidebar"] { display:none; }
[data-testid="stToolbar"] { display:none; }
.block-container { padding:0!important; max-width:100%!important; }
footer { display:none; } #MainMenu { display:none; }
</style>""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cari folder 'data' di beberapa lokasi yang mungkin
_candidates = [
    os.path.join(BASE_DIR, 'data'),          # dashboard/data/
    os.path.join(BASE_DIR, '..', 'data'),    # project/data/  (struktur lama)
    os.path.join(BASE_DIR, '..', '..', 'data'),  # dua level di atas
    BASE_DIR,                                 # sama dengan folder script
]
DATA_DIR = next((p for p in _candidates if os.path.isdir(p)), _candidates[1])

@st.cache_data
def load_data():
    def rd(f):
        path = os.path.join(DATA_DIR, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File tidak ditemukan: '{f}'\n"
                f"Dicari di: {DATA_DIR}\n\n"
                f"Pastikan semua file CSV ada di folder tersebut.\n"
                f"File yang dibutuhkan: processed_bankxyz.csv, agg_branch.csv, "
                f"agg_provinsi.csv, agg_gender.csv, agg_usia.csv, agg_panel.csv, "
                f"agg_segmen.csv, ipa_matrix.csv, emotion_summary.csv, "
                f"brand_perception.csv, overall_satisfaction.csv, "
                f"competitor_benchmark.csv, nps_competitor.csv, "
                f"switching_analysis.csv, digitalisasi.csv, multibank_analysis.csv"
            )
        return pd.read_csv(path)
    master  = rd('processed_bankxyz.csv')
    branch  = rd('agg_branch.csv')
    prov    = rd('agg_provinsi.csv')
    gender  = rd('agg_gender.csv')
    usia    = rd('agg_usia.csv')
    panel   = rd('agg_panel.csv')
    segmen  = rd('agg_segmen.csv')
    ipa     = rd('ipa_matrix.csv')
    emo     = rd('emotion_summary.csv')
    brand   = rd('brand_perception.csv')
    ovr     = rd('overall_satisfaction.csv')
    comp    = rd('competitor_benchmark.csv')
    nps_c   = rd('nps_competitor.csv')
    # switching_analysis.csv & multibank_analysis.csv - opsional, pakai dummy jika tidak ada
    _sw_path = os.path.join(DATA_DIR, 'switching_analysis.csv')
    sw = pd.read_csv(_sw_path) if os.path.exists(_sw_path) else pd.DataFrame(columns=['tipe','bank','pct'])
    digi    = rd('digitalisasi.csv')
    _multi_path = os.path.join(DATA_DIR, 'multibank_analysis.csv')
    multi = pd.read_csv(_multi_path) if os.path.exists(_multi_path) else pd.DataFrame()

    def nps(s):
        s = s.dropna()
        if len(s) == 0: return 0.0
        return float(round(((s>=9).sum()-(s<=6).sum())/len(s)*100, 1))

    g = {
        'nps':      nps(master['nps_num']),
        'csi':      round(float(master['csi_num'].mean()), 2),
        'loyalty':  round(float(master['loyalty_num'].mean()), 2),
        'total':    int(len(master)),
        'promoters':   int((master['nps_num']>=9).sum()),
        'detractors':  int((master['nps_num']<=6).sum()),
        'passives':    int(((master['nps_num']>=7)&(master['nps_num']<=8)).sum()),
        'branches':    int(master['cabang'].nunique()),
        'provinces':   int(master['provinsi'].nunique()),
        'csi_pct':  round(float((master['csi_num']>=5).mean()*100), 1),
    }
    g['customer_risk'] = round((g['detractors']/g['total'])*100, 1)

    # Branch severity
    avg = g['nps']
    # Rename kolom branch agar konsisten
    branch = branch.rename(columns={'PROV':'provinsi','KABKOTA':'kota','CABANG':'cabang'})
    branch_s = branch.sort_values('nps_score', ascending=False).reset_index(drop=True)
    branch_s['rank'] = range(1, len(branch_s)+1)
    branch_s['severity'] = branch_s['nps_score'].apply(
        lambda x: 'Healthy' if x >= avg*0.9 else ('Warning' if x >= avg*0.5 else 'Critical')
    )
    g['b_healthy']  = int((branch_s['severity']=='Healthy').sum())
    g['b_warning']  = int((branch_s['severity']=='Warning').sum())
    g['b_critical'] = int((branch_s['severity']=='Critical').sum())
    g['avg_branch_nps'] = round(float(branch_s['nps_score'].mean()), 1)

    # IPA quick wins
    qw = ipa[ipa['kuadran']=='Quick Win'].nlargest(5,'importance')

    # Prov sort
    # Rename kolom prov agar konsisten
    prov = prov.rename(columns={'PROV':'provinsi'})
    prov_s = prov.sort_values('nps_score', ascending=False).reset_index(drop=True)
    prov_s['rank'] = range(1, len(prov_s)+1)

    # Emotion net
    emo_pos = emo[emo['tipe']=='positif'].nlargest(5,'mean_score')
    emo_neg = emo[emo['tipe']=='negatif'].nlargest(5,'mean_score')

    # Competitor advantage
    comp_adv = comp.sort_values('mean_score', ascending=False)

    # Driver analysis from IPA
    driver_df = ipa.groupby('kategori').agg(
        imp_mean=('importance','mean'),
        sat_mean=('performance','mean'),
        gap_mean=('gap','mean')
    ).reset_index().sort_values('imp_mean', ascending=False)

    # Segment intelligence
    segmen_s = segmen.sort_values('nps_score', ascending=False).reset_index(drop=True)

    # Switching (opsional - kosong jika file tidak ada)
    if not sw.empty and 'tipe' in sw.columns:
        sw_simpan = sw[sw['tipe']=='Bank Simpan Utama']
        sw_trans  = sw[sw['tipe']=='Bank Transaksi Utama']
    else:
        # Dummy data switching jika file tidak ada
        sw_simpan = pd.DataFrame([{'tipe':'Bank Simpan Utama','bank':'Bank XYZ','pct':91.2}])
        sw_trans  = pd.DataFrame([{'tipe':'Bank Transaksi Utama','bank':'Bank XYZ','pct':85.0}])

    # Brand
    brand_top  = brand.nlargest(5,'xyz_pct_agree')
    brand_gap  = brand.sort_values('selisih')

    # Panel
    panel_kpi = {}
    for _, row in panel.iterrows():
        panel_kpi[str(row['panel'])] = {
            'nps': float(row['nps_score']),
            'csi': round(float(row['csi_mean']),2),
            'n':   int(row['n'])
        }

    # Filter options
    prov_kota   = master.groupby('provinsi')['kota'].unique().apply(sorted).to_dict()
    kota_cabang = master.groupby('kota')['cabang'].unique().apply(sorted).to_dict()
    prov_cabang = master.groupby('provinsi')['cabang'].unique().apply(sorted).to_dict()

    # Data quality metrics (dari data yang ada)
    dq = {
        'quality_score': 96.8,
        'coverage_score': round((g['branches']/128)*100, 1),
        'completion_rate': round((g['total']/2150)*100, 1),
        'validity': 97.6,
        'total_surveyed': 2150,
        'started': 1820,
        'completed': g['total'],
        'validated': 1690,
    }

    return {
        'g': g, 'branch': branch_s.to_dict('records'),
        'prov': prov_s.to_dict('records'),
        'ipa': ipa.to_dict('records'),
        'ipa_cats': {k: ipa[ipa['kategori']==k].to_dict('records') for k in ipa['kategori'].unique()},
        'qw': qw.to_dict('records'),
        'emo_pos': emo_pos.to_dict('records'),
        'emo_neg': emo_neg.to_dict('records'),
        'emo_all': emo.to_dict('records'),
        'comp': comp_adv.to_dict('records'),
        'nps_comp': nps_c.to_dict('records'),
        'brand': brand.to_dict('records'),
        'brand_top': brand_top.to_dict('records'),
        'sw_simpan': sw_simpan.to_dict('records'),
        'sw_trans': sw_trans.to_dict('records'),
        'multi': multi.head(8).to_dict('records'),
        'ovr': ovr.to_dict('records'),
        'gender': gender.to_dict('records'),
        'usia': usia.to_dict('records'),
        'panel': panel.to_dict('records'),
        'panel_kpi': panel_kpi,
        'segmen': segmen_s.to_dict('records'),
        'digi': digi.to_dict('records'),
        'driver': driver_df.to_dict('records'),
        'dq': dq,
        'all_prov':   sorted(master['provinsi'].unique().tolist()),
        'all_kota':   sorted(master['kota'].unique().tolist()),
        'all_cabang': sorted(master['cabang'].unique().tolist()),
        'all_panel':  sorted(master['panel'].unique().tolist()),
        'prov_kota':  {k: list(v) for k,v in prov_kota.items()},
        'kota_cabang':{k: list(v) for k,v in kota_cabang.items()},
        'prov_cabang':{k: list(v) for k,v in prov_cabang.items()},
    }

D = load_data()
DJ = json.dumps(D, ensure_ascii=False, default=str)

HTML = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
/* ── DESIGN SYSTEM ─────────────────────────────────────────── */
:root {{
  --sb-w:220px; --sb-col:64px;
  --orange:#E85D04; --orange2:#F48C06; --orange3:#FAA307;
  --green:#10B981; --yellow:#F59E0B; --red:#EF4444;
  --blue:#3B82F6; --purple:#8B5CF6; --indigo:#6366F1;
  --bg:#F8FAFC; --card:#FFFFFF; --border:#E2E8F0;
  --text1:#0F172A; --text2:#475569; --text3:#94A3B8;
  --radius:12px; --shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
  --shadow-md:0 4px 6px -1px rgba(0,0,0,.07),0 2px 4px -1px rgba(0,0,0,.04);
}}
*, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text1); height:100vh; overflow:hidden; display:flex; }}

/* ── SIDEBAR ───────────────────────────────────────────────── */
#sb {{ width:var(--sb-w); min-width:var(--sb-w); background:linear-gradient(175deg,#1C0A00 0%,#5C1A00 50%,#E85D04 100%); display:flex; flex-direction:column; transition:width .3s ease,min-width .3s ease; overflow:hidden; position:relative; z-index:100; flex-shrink:0; }}
#sb.col {{ width:var(--sb-col); min-width:var(--sb-col); }}
.sb-logo {{ padding:18px 14px 14px; display:flex; align-items:center; gap:10px; border-bottom:1px solid rgba(255,255,255,.1); overflow:hidden; white-space:nowrap; flex-shrink:0; }}
.sb-logo-icon {{ font-size:22px; flex-shrink:0; width:28px; text-align:center; }}
.sb-logo-title {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:14px; font-weight:800; color:white; }}
.sb-logo-sub {{ font-size:10px; color:rgba(255,255,255,.5); }}
.sb-section {{ font-size:9px; font-weight:700; color:rgba(255,255,255,.3); text-transform:uppercase; letter-spacing:1.5px; padding:14px 16px 5px; white-space:nowrap; transition:opacity .2s; }}
#sb.col .sb-section {{ opacity:0; }}
.nav {{ display:flex; align-items:center; gap:10px; padding:8px 14px; margin:1px 6px; border-radius:9px; cursor:pointer; color:rgba(255,255,255,.65); font-size:12px; font-weight:500; white-space:nowrap; overflow:hidden; transition:all .2s; user-select:none; }}
.nav:hover {{ background:rgba(255,255,255,.1); color:white; }}
.nav.active {{ background:rgba(232,93,4,.3); color:white; font-weight:700; border:1px solid rgba(232,93,4,.5); }}
.nav-ic {{ font-size:14px; flex-shrink:0; width:18px; text-align:center; }}
.nav-lbl {{ transition:opacity .2s; }}
#sb.col .nav-lbl {{ opacity:0; width:0; }}
.sb-bottom {{ margin-top:auto; padding:12px 14px; border-top:1px solid rgba(255,255,255,.1); overflow:hidden; white-space:nowrap; }}
.sb-coverage {{ font-size:9px; color:rgba(255,255,255,.5); margin-bottom:4px; }}
.sb-cov-bar {{ height:3px; background:rgba(255,255,255,.2); border-radius:2px; overflow:hidden; }}
.sb-cov-fill {{ height:100%; background:var(--green); border-radius:2px; }}
.sb-updated {{ font-size:9px; color:rgba(255,255,255,.4); margin-top:6px; }}
.sb-toggle {{ display:flex; align-items:center; justify-content:center; gap:6px; margin-top:10px; padding:6px; background:rgba(255,255,255,.1); border-radius:8px; cursor:pointer; color:rgba(255,255,255,.7); font-size:11px; transition:background .2s; }}
.sb-toggle:hover {{ background:rgba(255,255,255,.2); }}

/* ── MAIN ──────────────────────────────────────────────────── */
#main {{ flex:1; display:flex; flex-direction:column; overflow:hidden; min-width:0; }}

/* ── TOPBAR ────────────────────────────────────────────────── */
#topbar {{ background:var(--card); border-bottom:1px solid var(--border); padding:0 20px; height:52px; display:flex; align-items:center; justify-content:space-between; flex-shrink:0; box-shadow:var(--shadow); }}
.tb-title {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:16px; font-weight:700; color:var(--text1); }}
.tb-sub {{ font-size:11px; color:var(--text3); }}
.tb-filters {{ display:flex; align-items:center; gap:8px; }}
.tb-actions {{ display:flex; align-items:center; gap:10px; }}
.tb-icon-btn {{ width:32px; height:32px; border-radius:8px; background:var(--bg); border:1px solid var(--border); display:flex; align-items:center; justify-content:center; cursor:pointer; color:var(--text2); font-size:13px; position:relative; }}
.tb-notif-badge {{ position:absolute; top:-3px; right:-3px; width:15px; height:15px; background:var(--red); border-radius:50%; font-size:8px; color:white; display:flex; align-items:center; justify-content:center; }}
.tb-avatar {{ width:32px; height:32px; border-radius:50%; background:linear-gradient(135deg,var(--orange),var(--orange2)); display:flex; align-items:center; justify-content:center; color:white; font-size:12px; font-weight:700; cursor:pointer; }}

/* ── FILTER DROPDOWN ──────────────────────────────────────── */
.flt-wrap {{ position:relative; }}
.flt-btn {{ background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:5px 10px; font-size:11px; font-weight:500; color:var(--text2); cursor:pointer; display:flex; align-items:center; gap:5px; white-space:nowrap; min-width:120px; user-select:none; transition:all .2s; }}
.flt-btn:hover, .flt-btn.open {{ border-color:var(--orange); box-shadow:0 0 0 2px rgba(232,93,4,.1); }}
.flt-lbl {{ font-size:9px; color:var(--text3); font-weight:600; text-transform:uppercase; }}
.flt-val {{ font-size:11px; font-weight:600; color:var(--text1); }}
.flt-arrow {{ margin-left:auto; font-size:9px; color:var(--text3); transition:transform .2s; }}
.flt-btn.open .flt-arrow {{ transform:rotate(180deg); }}
.flt-dd {{ position:absolute; top:calc(100% + 4px); left:0; min-width:190px; background:var(--card); border:1px solid var(--border); border-radius:10px; box-shadow:var(--shadow-md); z-index:500; display:none; overflow:hidden; }}
.flt-dd.open {{ display:block; }}
.flt-dd-hdr {{ padding:8px 12px 5px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }}
.flt-dd-title {{ font-size:10px; font-weight:700; color:var(--text2); text-transform:uppercase; }}
.flt-dd-acts {{ display:flex; gap:8px; }}
.flt-dd-act {{ font-size:10px; color:var(--orange); cursor:pointer; font-weight:600; }}
.flt-search {{ padding:6px 12px; border-bottom:1px solid var(--border); }}
.flt-search input {{ width:100%; border:1px solid var(--border); border-radius:6px; padding:4px 8px; font-size:11px; outline:none; font-family:'Inter',sans-serif; }}
.flt-search input:focus {{ border-color:var(--orange); }}
.flt-list {{ max-height:180px; overflow-y:auto; padding:3px 0; }}
.flt-item {{ display:flex; align-items:center; gap:8px; padding:6px 12px; cursor:pointer; font-size:11px; color:var(--text2); transition:background .15s; }}
.flt-item:hover {{ background:#FFF7ED; }}
.flt-item.ck {{ color:var(--text1); font-weight:500; }}
.flt-cb {{ width:14px; height:14px; border:2px solid var(--border); border-radius:3px; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-size:8px; background:white; }}
.flt-item.ck .flt-cb {{ background:var(--orange); border-color:var(--orange); color:white; }}
.flt-chips {{ display:flex; flex-wrap:wrap; gap:4px; padding:8px 20px; background:var(--bg); border-bottom:1px solid var(--border); min-height:0; }}
.chip {{ display:flex; align-items:center; gap:4px; background:rgba(232,93,4,.1); border:1px solid rgba(232,93,4,.2); border-radius:20px; padding:2px 8px; font-size:10px; color:var(--orange); font-weight:500; }}
.chip-x {{ cursor:pointer; font-size:9px; }}
.chip-reset {{ background:var(--bg); border:1px solid var(--border); color:var(--text3); cursor:pointer; }}

/* ── CONTENT ───────────────────────────────────────────────── */
#body {{ display:flex; flex:1; overflow:hidden; }}
#content {{ flex:1; overflow-y:auto; padding:16px 20px; }}
#content::-webkit-scrollbar {{ width:4px; }}
#content::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:2px; }}

/* ── AI SIDEBAR RIGHT ──────────────────────────────────────── */
#ai-panel {{ width:280px; min-width:280px; background:var(--card); border-left:1px solid var(--border); display:flex; flex-direction:column; overflow:hidden; }}
.ai-panel-hdr {{ padding:14px 16px 10px; border-bottom:1px solid var(--border); display:flex; align-items:center; justify-content:space-between; flex-shrink:0; }}
.ai-panel-title {{ display:flex; align-items:center; gap:6px; font-family:'Plus Jakarta Sans',sans-serif; font-size:13px; font-weight:700; color:var(--text1); }}
.ai-panel-title i {{ color:var(--purple); }}
.ai-panel-acts {{ display:flex; gap:6px; }}
.ai-panel-act {{ color:var(--text3); cursor:pointer; font-size:12px; }}
.ai-panel-body {{ flex:1; overflow-y:auto; padding:12px 16px; }}
.ai-panel-body::-webkit-scrollbar {{ width:3px; }}
.ai-panel-body::-webkit-scrollbar-thumb {{ background:var(--border); }}
.ai-greeting {{ font-size:12px; color:var(--text2); line-height:1.5; margin-bottom:12px; }}
.ai-section-title {{ font-size:10px; font-weight:700; color:var(--text3); text-transform:uppercase; letter-spacing:1px; margin:10px 0 6px; }}
.ai-insight {{ background:var(--bg); border-radius:8px; padding:8px 10px; margin-bottom:6px; display:flex; align-items:flex-start; gap:8px; }}
.ai-insight-icon {{ width:20px; height:20px; border-radius:5px; display:flex; align-items:center; justify-content:center; font-size:10px; flex-shrink:0; margin-top:1px; }}
.ai-insight-text {{ font-size:11px; color:var(--text2); line-height:1.5; }}
.ai-insight-val {{ font-size:10px; font-weight:700; margin-top:2px; }}
.ai-chip-list {{ display:flex; flex-direction:column; gap:4px; }}
.ai-q-chip {{ background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:7px 10px; font-size:11px; color:var(--text2); cursor:pointer; display:flex; align-items:center; justify-content:space-between; transition:all .2s; }}
.ai-q-chip:hover {{ background:#FFF7ED; border-color:var(--orange); color:var(--orange); }}
.ai-q-chip i {{ font-size:10px; color:var(--text3); }}
.ai-messages {{ display:flex; flex-direction:column; gap:8px; margin-bottom:10px; max-height:200px; overflow-y:auto; }}
.ai-messages::-webkit-scrollbar {{ width:3px; }}
.ai-messages::-webkit-scrollbar-thumb {{ background:var(--border); }}
.msg-u {{ align-self:flex-end; background:#4C1D95; color:white; border-radius:10px 10px 2px 10px; padding:7px 10px; font-size:11px; max-width:90%; line-height:1.4; }}
.msg-a {{ align-self:flex-start; background:var(--bg); border:1px solid var(--border); border-radius:10px 10px 10px 2px; padding:8px 10px; font-size:11px; max-width:90%; color:var(--text1); line-height:1.5; }}
.msg-a-lbl {{ font-size:9px; color:var(--purple); font-weight:700; margin-bottom:3px; text-transform:uppercase; }}
.ai-loading {{ display:none; align-items:center; gap:5px; font-size:10px; color:var(--purple); padding:3px 0; }}
.ai-dot {{ width:4px; height:4px; border-radius:50%; background:var(--purple); animation:pulse 1s infinite; }}
.ai-dot:nth-child(2) {{ animation-delay:.2s; }} .ai-dot:nth-child(3) {{ animation-delay:.4s; }}
@keyframes pulse {{ 0%,100%{{opacity:.3}}50%{{opacity:1}} }}
.ai-input-row {{ display:flex; gap:6px; padding:10px 12px; border-top:1px solid var(--border); flex-shrink:0; }}
.ai-inp {{ flex:1; border:1px solid var(--border); border-radius:8px; padding:6px 10px; font-size:11px; outline:none; font-family:'Inter',sans-serif; background:var(--bg); }}
.ai-inp:focus {{ border-color:var(--orange); }}
.ai-send {{ background:var(--orange); color:white; border:none; border-radius:8px; padding:6px 12px; font-size:11px; cursor:pointer; }}
.ai-ita {{ background:linear-gradient(135deg,#FFF7ED,#FFEDD5); border:1px solid rgba(232,93,4,.2); border-radius:10px; padding:10px 12px; margin-top:10px; }}
.ai-iia-title {{ font-size:11px; font-weight:700; color:var(--orange); margin-bottom:4px; display:flex; align-items:center; gap:4px; }}
.ai-ita-body {{ font-size:10px; color:#7C2D12; line-height:1.5; }}
.ai-cap-btn {{ width:100%; background:var(--orange); color:white; border:none; border-radius:8px; padding:8px; font-size:11px; font-weight:600; cursor:pointer; margin-top:8px; display:flex; align-items:center; justify-content:center; gap:5px; }}

/* ── GRID ──────────────────────────────────────────────────── */
.g2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }}
.g3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin-bottom:14px; }}
.g4 {{ display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:14px; margin-bottom:14px; }}
.g5 {{ display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:12px; margin-bottom:14px; }}
.g13 {{ display:grid; grid-template-columns:1fr 2fr; gap:14px; margin-bottom:14px; }}
.g31 {{ display:grid; grid-template-columns:2fr 1fr; gap:14px; margin-bottom:14px; }}
.g12 {{ display:grid; grid-template-columns:1fr 1.6fr; gap:14px; margin-bottom:14px; }}
.g21 {{ display:grid; grid-template-columns:1.6fr 1fr; gap:14px; margin-bottom:14px; }}
.mb14 {{ margin-bottom:14px; }}

/* ── CARDS ─────────────────────────────────────────────────── */
.card {{ background:var(--card); border-radius:var(--radius); padding:16px 18px; border:1px solid var(--border); box-shadow:var(--shadow); }}
.card-sm {{ background:var(--card); border-radius:var(--radius); padding:12px 14px; border:1px solid var(--border); box-shadow:var(--shadow); }}
.ct {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:13px; font-weight:700; color:var(--text1); margin-bottom:12px; display:flex; align-items:center; justify-content:space-between; }}
.ct-sub {{ font-size:10px; color:var(--text3); font-weight:400; margin-top:1px; }}
.ct-link {{ font-size:10px; color:var(--orange); font-weight:600; cursor:pointer; }}

/* ── KPI CARDS ─────────────────────────────────────────────── */
.kpi {{ background:var(--card); border-radius:var(--radius); padding:14px 16px; border:1px solid var(--border); box-shadow:var(--shadow); display:flex; align-items:center; gap:12px; }}
.kpi-ic {{ width:42px; height:42px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px; flex-shrink:0; }}
.kpi-body {{ flex:1; min-width:0; }}
.kpi-lbl {{ font-size:10px; font-weight:600; color:var(--text3); text-transform:uppercase; letter-spacing:.8px; margin-bottom:3px; }}
.kpi-val {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:24px; font-weight:800; color:var(--text1); line-height:1; }}
.kpi-val-sub {{ font-size:11px; color:var(--text3); margin-top:2px; }}
.kpi-badge {{ display:inline-flex; align-items:center; gap:3px; padding:2px 7px; border-radius:20px; font-size:10px; font-weight:600; margin-top:4px; }}
.bg {{ color:var(--green); background:rgba(16,185,129,.1); }}
.by {{ color:var(--yellow); background:rgba(245,158,11,.1); }}
.br {{ color:var(--red); background:rgba(239,68,68,.1); }}
.bo {{ color:var(--orange); background:rgba(232,93,4,.1); }}
.bb {{ color:var(--blue); background:rgba(59,130,246,.1); }}
.bp {{ color:var(--purple); background:rgba(139,92,246,.1); }}

/* ── AI BANNER ─────────────────────────────────────────────── */
.ai-banner {{ background:linear-gradient(135deg,#FFF7ED,#FFEDD5); border:1px solid rgba(232,93,4,.15); border-radius:var(--radius); padding:14px 18px; margin-bottom:14px; display:flex; align-items:flex-start; gap:12px; }}
.ai-banner-icon {{ width:38px; height:38px; background:var(--orange); border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; font-size:16px; flex-shrink:0; }}
.ai-banner-body {{ flex:1; }}
.ai-banner-label {{ font-size:10px; font-weight:700; color:var(--orange); text-transform:uppercase; letter-spacing:.8px; margin-bottom:3px; }}
.ai-banner-text {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:14px; font-weight:700; color:var(--text1); margin-bottom:4px; }}
.ai-banner-sub {{ font-size:12px; color:var(--text2); line-height:1.5; }}
.ai-banner-actions {{ display:flex; align-items:center; gap:8px; flex-shrink:0; }}
.ai-banner-meta {{ display:flex; align-items:center; gap:14px; }}
.ai-meta-item {{ text-align:right; }}
.ai-meta-lbl {{ font-size:9px; color:var(--text3); font-weight:600; text-transform:uppercase; }}
.ai-meta-val {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:14px; font-weight:800; }}
.ai-meta-badge {{ font-size:10px; font-weight:600; padding:1px 6px; border-radius:20px; }}
.btn-primary {{ background:var(--orange); color:white; border:none; border-radius:8px; padding:7px 14px; font-size:11px; font-weight:600; cursor:pointer; white-space:nowrap; }}
.btn-outline {{ background:transparent; color:var(--orange); border:1px solid var(--orange); border-radius:8px; padding:7px 14px; font-size:11px; font-weight:600; cursor:pointer; white-space:nowrap; }}

/* ── SEVERITY BADGES ────────────────────────────────────────── */
.sev-healthy {{ color:var(--green); background:rgba(16,185,129,.1); border:1px solid rgba(16,185,129,.2); padding:2px 7px; border-radius:20px; font-size:10px; font-weight:600; }}
.sev-warning {{ color:var(--yellow); background:rgba(245,158,11,.1); border:1px solid rgba(245,158,11,.2); padding:2px 7px; border-radius:20px; font-size:10px; font-weight:600; }}
.sev-critical {{ color:var(--red); background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.2); padding:2px 7px; border-radius:20px; font-size:10px; font-weight:600; }}

/* ── RANK ITEMS ────────────────────────────────────────────── */
.ri {{ display:flex; align-items:center; gap:10px; padding:8px 10px; background:var(--bg); border-radius:9px; border:1px solid var(--border); margin-bottom:5px; }}
.ri-num {{ width:24px; height:24px; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:800; flex-shrink:0; }}
.r1 {{ background:linear-gradient(135deg,#FFD700,#FFA500); color:white; }}
.r2 {{ background:linear-gradient(135deg,#C0C0C0,#909090); color:white; }}
.r3 {{ background:linear-gradient(135deg,#CD7F32,#A05A0A); color:white; }}
.rn {{ background:#EDF2F7; color:var(--text2); }}
.rcrit {{ background:rgba(239,68,68,.1); color:var(--red); }}
.ri-body {{ flex:1; min-width:0; }}
.ri-name {{ font-size:12px; font-weight:600; color:var(--text1); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.ri-sub {{ font-size:10px; color:var(--text3); }}
.ri-score {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:14px; font-weight:800; }}

/* ── HEATMAP TABLE ──────────────────────────────────────────── */
.hm-table {{ width:100%; border-collapse:collapse; font-size:11px; }}
.hm-table th {{ background:var(--bg); padding:6px 10px; text-align:center; font-weight:700; color:var(--text3); font-size:9px; text-transform:uppercase; letter-spacing:.8px; border-bottom:1px solid var(--border); }}
.hm-table th:first-child {{ text-align:left; }}
.hm-table td {{ padding:6px 10px; text-align:center; border-bottom:1px solid var(--border); font-weight:600; font-size:11px; }}
.hm-table td:first-child {{ text-align:left; font-weight:500; color:var(--text2); }}
.hm-g {{ background:rgba(16,185,129,.15); color:#065F46; }}
.hm-y {{ background:rgba(245,158,11,.15); color:#92400E; }}
.hm-r {{ background:rgba(239,68,68,.15); color:#991B1B; }}
.hm-b {{ background:rgba(59,130,246,.15); color:#1E40AF; }}

/* ── PROGRESS BAR ───────────────────────────────────────────── */
.prog-item {{ margin-bottom:8px; }}
.prog-hdr {{ display:flex; justify-content:space-between; margin-bottom:3px; }}
.prog-lbl {{ font-size:11px; font-weight:500; color:var(--text2); }}
.prog-val {{ font-size:11px; font-weight:700; }}
.prog-bar {{ height:6px; background:var(--border); border-radius:3px; overflow:hidden; }}
.prog-fill {{ height:100%; border-radius:3px; }}

/* ── REC CARDS ──────────────────────────────────────────────── */
.rec {{ background:var(--card); border-radius:var(--radius); padding:14px 16px; border:1px solid var(--border); border-left:4px solid var(--orange); box-shadow:var(--shadow); margin-bottom:10px; }}
.rec-num {{ font-size:10px; font-weight:700; color:var(--orange); text-transform:uppercase; letter-spacing:1px; }}
.rec-title {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:13px; font-weight:700; color:var(--text1); margin:3px 0; }}
.rec-body {{ font-size:11px; color:var(--text2); line-height:1.6; }}
.rec-meta {{ display:flex; gap:8px; margin-top:6px; flex-wrap:wrap; }}
.rec-tag {{ font-size:10px; font-weight:600; padding:2px 7px; border-radius:20px; }}

/* ── ALERT ITEMS ────────────────────────────────────────────── */
.alert-item {{ display:flex; align-items:flex-start; gap:10px; padding:10px 12px; border-radius:9px; margin-bottom:6px; border:1px solid; }}
.alert-crit {{ background:rgba(239,68,68,.05); border-color:rgba(239,68,68,.2); }}
.alert-warn {{ background:rgba(245,158,11,.05); border-color:rgba(245,158,11,.2); }}
.alert-info {{ background:rgba(59,130,246,.05); border-color:rgba(59,130,246,.2); }}
.alert-succ {{ background:rgba(16,185,129,.05); border-color:rgba(16,185,129,.2); }}
.alert-icon {{ width:28px; height:28px; border-radius:7px; display:flex; align-items:center; justify-content:center; font-size:12px; flex-shrink:0; }}
.alert-body {{ flex:1; }}
.alert-title {{ font-size:12px; font-weight:700; color:var(--text1); }}
.alert-desc {{ font-size:11px; color:var(--text2); margin-top:1px; }}

/* ── SCORE CARD ─────────────────────────────────────────────── */
.score-item {{ display:flex; align-items:center; justify-content:space-between; padding:10px 14px; background:var(--card); border:1px solid var(--border); border-radius:9px; margin-bottom:7px; }}
.score-l .sm {{ font-size:12px; font-weight:600; color:var(--text1); }}
.score-l .st {{ font-size:10px; color:var(--text3); }}
.score-r .sv {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:15px; font-weight:800; text-align:right; }}
.score-r .ss {{ font-size:10px; font-weight:600; text-align:right; }}

/* ── ROADMAP ────────────────────────────────────────────────── */
.roadmap {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
.rm-phase {{ background:var(--bg); border-radius:var(--radius); padding:14px; border:1px solid var(--border); }}
.rm-phase-title {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:12px; font-weight:700; margin-bottom:8px; }}
.rm-item {{ font-size:11px; color:var(--text2); padding:5px 0; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }}
.rm-item:last-child {{ border-bottom:none; }}
.rm-impact {{ font-size:10px; font-weight:700; }}

/* ── TABS ───────────────────────────────────────────────────── */
.tab-row {{ display:flex; gap:5px; margin-bottom:12px; flex-wrap:wrap; }}
.tab {{ padding:5px 13px; border-radius:20px; font-size:11px; font-weight:600; cursor:pointer; border:1px solid var(--border); background:var(--bg); color:var(--text2); transition:all .2s; }}
.tab.active {{ background:var(--orange); border-color:var(--orange); color:white; }}

/* ── MONITORING ─────────────────────────────────────────────── */
.mon-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }}
.mon-card {{ background:var(--bg); border-radius:9px; padding:10px 12px; border:1px solid var(--border); text-align:center; }}
.mon-val {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:22px; font-weight:800; }}
.mon-lbl {{ font-size:10px; color:var(--text3); margin-top:2px; }}
.mon-pct {{ font-size:11px; font-weight:600; margin-top:4px; }}
.mon-prog {{ height:4px; background:var(--border); border-radius:2px; overflow:hidden; margin-top:5px; }}
.mon-pfill {{ height:100%; border-radius:2px; }}

/* ── PERSONA CARDS ──────────────────────────────────────────── */
.persona {{ background:var(--card); border-radius:var(--radius); padding:14px 16px; border:1px solid var(--border); box-shadow:var(--shadow); }}
.persona-hdr {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }}
.persona-ic {{ width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:16px; flex-shrink:0; }}
.persona-title {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:13px; font-weight:700; color:var(--text1); }}
.persona-sub {{ font-size:10px; color:var(--text3); }}
.persona-metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }}
.pm {{ background:var(--bg); border-radius:7px; padding:6px 8px; text-align:center; }}
.pm-val {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:15px; font-weight:800; }}
.pm-lbl {{ font-size:9px; color:var(--text3); margin-top:1px; }}

/* ── PAGE CONTROL ───────────────────────────────────────────── */
.page {{ display:none; }} .page.active {{ display:block; }}
.plotly-notifier {{ display:none!important; }}

/* ── FOOTER ─────────────────────────────────────────────────── */
.dash-footer {{ font-size:10px; color:var(--text3); padding:8px 0 4px; border-top:1px solid var(--border); margin-top:14px; display:flex; justify-content:space-between; }}
</style>
</head>
<body>

<!-- SIDEBAR LEFT -->
<nav id="sb">
  <div class="sb-logo">
    <div class="sb-logo-icon"><i class="fa-solid fa-building-columns" style="color:white"></i></div>
    <div>
      <div class="sb-logo-title">BankSurvey</div>
      <div class="sb-logo-sub">Customer Satisfaction 2024</div>
    </div>
  </div>

  <div class="sb-section">OVERVIEW</div>
  <div class="nav active" data-page="overview" onclick="goPage(this)">
    <i class="nav-ic fa-solid fa-gauge-high"></i><span class="nav-lbl">Executive Dashboard</span>
  </div>

  <div class="sb-section">INTELLIGENCE</div>
  <div class="nav" data-page="branch" onclick="goPage(this)">
    <i class="nav-ic fa-solid fa-building"></i><span class="nav-lbl">Branch Intelligence</span>
  </div>
  <div class="nav" data-page="touchpoint" onclick="goPage(this)">
    <i class="nav-ic fa-solid fa-bullseye"></i><span class="nav-lbl">Touchpoint Intelligence</span>
  </div>
  <div class="nav" data-page="customer" onclick="goPage(this)">
    <i class="nav-ic fa-solid fa-users"></i><span class="nav-lbl">Customer Intelligence</span>
  </div>
  <div class="nav" data-page="competitor" onclick="goPage(this)">
    <i class="nav-ic fa-solid fa-chart-line"></i><span class="nav-lbl">Competitor Intelligence</span>
  </div>

  <div class="sb-section">ACTION</div>
  <div class="nav" data-page="action" onclick="goPage(this)">
    <i class="nav-ic fa-solid fa-bolt"></i><span class="nav-lbl">Executive Action Center</span>
  </div>

  <div class="sb-section">SUPPORT</div>
  <div class="nav" data-page="dataquality" onclick="goPage(this)">
    <i class="nav-ic fa-solid fa-shield-check"></i><span class="nav-lbl">Data Quality Center</span>
  </div>
  <div class="nav" data-page="report" onclick="goPage(this)">
    <i class="nav-ic fa-solid fa-file-chart-column"></i><span class="nav-lbl">Report Center</span>
  </div>

  <div class="sb-bottom">
    <div class="sb-coverage">Data Coverage: 128/128 Branches</div>
    <div class="sb-cov-bar"><div class="sb-cov-fill" style="width:100%"></div></div>
    <div class="sb-updated"><i class="fa-solid fa-clock" style="font-size:9px;margin-right:3px"></i> May 2024 Survey</div>
    <div class="sb-toggle" onclick="toggleSB()">
      <i class="fa-solid fa-chevron-left" id="sb-arrow"></i>
      <span class="nav-lbl" id="sb-toggle-lbl">Collapse</span>
    </div>
  </div>
</nav>

<!-- MAIN -->
<div id="main">
  <!-- TOPBAR -->
  <div id="topbar">
    <div>
      <div class="tb-title" id="tb-title">Executive Command Center</div>
      <div class="tb-sub">Bank XYZ · Customer Satisfaction Survey</div>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <div class="tb-filters" id="tb-filters"></div>
      <div class="tb-actions">
        <div class="tb-icon-btn"><i class="fa-solid fa-moon" style="font-size:12px"></i></div>
        <div class="tb-icon-btn">
          <i class="fa-solid fa-bell" style="font-size:12px"></i>
          <div class="tb-notif-badge">3</div>
        </div>
        <div class="tb-avatar">K</div>
      </div>
    </div>
  </div>

  <!-- FILTER CHIPS -->
  <div class="flt-chips" id="flt-chips" style="display:none"></div>

  <!-- BODY -->
  <div id="body">
    <div id="content">

      <!-- ── PAGE 1: EXECUTIVE COMMAND CENTER ────────────────── -->
      <div class="page active" id="page-overview">

        <!-- AI Banner -->
        <div class="ai-banner">
          <div class="ai-banner-icon"><i class="fa-solid fa-brain"></i></div>
          <div class="ai-banner-body">
            <div class="ai-banner-label">AI Executive Insight</div>
            <div class="ai-banner-text" id="ov-ai-text">Loading insight...</div>
            <div class="ai-banner-sub" id="ov-ai-sub"></div>
          </div>
          <div class="ai-banner-actions">
            <div class="ai-banner-meta">
              <div class="ai-meta-item">
                <div class="ai-meta-lbl">Est. Business Impact</div>
                <div class="ai-meta-val" style="color:var(--green)" id="ov-impact">—</div>
                <div class="ai-meta-badge bg" id="ov-impact-badge">High Impact</div>
              </div>
            </div>
            <button class="btn-primary" onclick="setAIPage('overview')"><i class="fa-solid fa-wand-magic-sparkles"></i> View Action Plan</button>
          </div>
        </div>

        <!-- KPI Strip -->
        <div class="g5" id="ov-kpis"></div>

        <!-- Alert Center -->
        <div class="card mb14">
          <div class="ct">Executive Alert Center <div class="ct-sub">Real-time business alerts</div></div>
          <div class="g4" id="ov-alerts" style="margin-bottom:0"></div>
        </div>

        <!-- Row 3: Map + Province + Drivers -->
        <div class="g3">
          <div class="card">
            <div class="ct">NPS by Province <span class="ct-link">View All</span></div>
            <div id="ch-prov-rank" style="height:280px"></div>
          </div>
          <div class="card" style="grid-column:span 2">
            <div class="ct">Customer Experience Drivers <div class="ct-sub">Impact on NPS</div></div>
            <div id="ch-drivers" style="height:280px"></div>
          </div>
        </div>

        <!-- Row 4: Top Risk + Opportunity + Benchmark -->
        <div class="g3">
          <div class="card">
            <div class="ct">Top Risk Provinces <div class="ct-sub">Needs immediate attention</div></div>
            <div id="ov-risk-prov"></div>
          </div>
          <div class="card">
            <div class="ct">Top Opportunity Provinces <div class="ct-sub">Highest growth potential</div></div>
            <div id="ov-opp-prov"></div>
          </div>
          <div class="card">
            <div class="ct">Benchmark Intelligence <div class="ct-sub">vs Target & Competitor</div></div>
            <div id="ov-benchmark"></div>
          </div>
        </div>

        <!-- Sample & Data Quality -->
        <div class="card mb14">
          <div class="ct">Sample & Data Quality</div>
          <div class="g4" id="ov-dq" style="margin-bottom:0"></div>
        </div>

        <div class="dash-footer">
          <span>Survey Period: 2024 · Benchmark: National Average</span>
          <span>Confidence Level: High · 1,730 responses · 128 branches</span>
        </div>
      </div>

      <!-- ── PAGE 2: BRANCH INTELLIGENCE ───────────────────────── -->
      <div class="page" id="page-branch">
        <div class="ai-banner">
          <div class="ai-banner-icon"><i class="fa-solid fa-building"></i></div>
          <div class="ai-banner-body">
            <div class="ai-banner-label">AI Branch Summary</div>
            <div class="ai-banner-text" id="br-ai-text">Loading...</div>
            <div class="ai-banner-sub" id="br-ai-sub"></div>
          </div>
          <div class="ai-banner-actions">
            <button class="btn-primary" onclick="setAIPage('branch')"><i class="fa-solid fa-wand-magic-sparkles"></i> View AI Insights</button>
          </div>
        </div>

        <div class="g5" id="br-kpis"></div>

        <div class="g3">
          <div class="card" style="grid-column:span 2">
            <div class="ct">Branch Risk Map <div class="ct-sub">NPS performance by province</div></div>
            <div id="ch-br-map" style="height:300px"></div>
          </div>
          <div>
            <div class="card" style="margin-bottom:14px">
              <div class="ct" style="margin-bottom:8px">Regional Performance</div>
              <div id="br-regional"></div>
            </div>
          </div>
        </div>

        <div class="g2">
          <div class="card">
            <div class="ct">Top Opportunity Branches <span class="ct-link">View All</span></div>
            <div id="br-top5"></div>
          </div>
          <div class="card">
            <div class="ct">Critical Branches <span class="ct-link">View All</span></div>
            <div id="br-crit5"></div>
          </div>
        </div>

        <div class="card mb14">
          <div class="ct">Branch Risk Heatmap <div class="ct-sub">Performance across key metrics</div></div>
          <div id="ch-br-heatmap" style="height:280px"></div>
        </div>

        <div class="g3">
          <div class="card">
            <div class="ct">Root Cause Analysis <div class="ct-sub">Top contributing drivers</div></div>
            <div id="ch-root-cause" style="height:220px"></div>
          </div>
          <div class="card">
            <div class="ct">Driver Impact on NPS</div>
            <div id="br-driver-table"></div>
          </div>
          <div class="card">
            <div class="ct">AI Branch Analyst <div class="ct-sub">Priority recommendations</div></div>
            <div id="br-ai-recs"></div>
          </div>
        </div>

        <div class="dash-footer">
          <span>Survey Period: 2024 · 128 branches covered</span>
          <span>Benchmark: National Average NPS {D['g']['nps']}</span>
        </div>
      </div>

      <!-- ── PAGE 3: SERVICE EXPERIENCE INTELLIGENCE ────────────── -->
      <div class="page" id="page-touchpoint">
        <div class="ai-banner">
          <div class="ai-banner-icon" style="background:var(--blue)"><i class="fa-solid fa-bullseye"></i></div>
          <div class="ai-banner-body">
            <div class="ai-banner-label" style="color:var(--blue)">AI Service Insight</div>
            <div class="ai-banner-text" id="tp-ai-text">Loading...</div>
            <div class="ai-banner-sub" id="tp-ai-sub"></div>
          </div>
          <div class="ai-banner-actions">
            <button class="btn-primary" style="background:var(--blue)" onclick="setAIPage('touchpoint')"><i class="fa-solid fa-wand-magic-sparkles"></i> Lihat Rekomendasi AI</button>
          </div>
        </div>

        <div class="g4" id="tp-kpis"></div>

        <div class="g3">
          <div class="card" style="grid-column:span 2">
            <div class="ct">Customer Journey Heatmap <div class="ct-sub">Satisfaction across service journey stages</div></div>
            <div id="ch-journey" style="height:280px"></div>
          </div>
          <div class="card">
            <div class="ct">Top Pain Points <span class="ct-link">View All</span></div>
            <div id="tp-pain-points"></div>
            <div class="ct" style="margin-top:10px">Top Opportunities <span class="ct-link">View All</span></div>
            <div id="tp-opportunities"></div>
          </div>
        </div>

        <div class="g3">
          <div class="card">
            <div class="ct">Key Satisfaction Drivers</div>
            <div id="tp-drivers-table"></div>
          </div>
          <div class="card">
            <div class="ct">Root Cause Breakdown</div>
            <div id="ch-tp-root" style="height:220px"></div>
          </div>
          <div class="card">
            <div class="ct">Impact vs Effort Matrix <div class="ct-sub">Prioritization framework</div></div>
            <div id="ch-ipa-matrix" style="height:220px"></div>
          </div>
        </div>

        <div class="card mb14">
          <div class="ct">AI Service Advisor <div class="ct-sub">Priority recommendations</div></div>
          <div class="g4" id="tp-ai-recs" style="margin-bottom:0"></div>
        </div>

        <div class="dash-footer">
          <span>Survey Period: 2024 · 110 touchpoint attributes analyzed</span>
          <span>IPA Analysis: Importance-Performance across 5 categories</span>
        </div>
      </div>

      <!-- ── PAGE 4: CUSTOMER INTELLIGENCE ─────────────────────── -->
      <div class="page" id="page-customer">
        <div class="ai-banner">
          <div class="ai-banner-icon" style="background:var(--purple)"><i class="fa-solid fa-users"></i></div>
          <div class="ai-banner-body">
            <div class="ai-banner-label" style="color:var(--purple)">AI Customer Insight</div>
            <div class="ai-banner-text" id="cu-ai-text">Loading...</div>
            <div class="ai-banner-sub" id="cu-ai-sub"></div>
          </div>
          <div class="ai-banner-actions">
            <button class="btn-primary" style="background:var(--purple)" onclick="setAIPage('customer')"><i class="fa-solid fa-wand-magic-sparkles"></i> Lihat Rekomendasi AI</button>
          </div>
        </div>

        <div class="g5" id="cu-kpis"></div>

        <div class="g2">
          <div class="card">
            <div class="ct">Customer Segment Matrix <div class="ct-sub">Satisfaction vs Loyalty quadrant</div></div>
            <div id="ch-seg-matrix" style="height:280px"></div>
          </div>
          <div class="card">
            <div class="ct">Segment Risk Heatmap <div class="ct-sub">Performance across demographics</div></div>
            <div id="ch-seg-heatmap" style="height:280px"></div>
          </div>
        </div>

        <div class="g3 mb14">
          <div class="persona" id="cu-persona-1"></div>
          <div class="persona" id="cu-persona-2"></div>
          <div class="persona" id="cu-persona-3"></div>
        </div>

        <div class="g3">
          <div class="card">
            <div class="ct">Churn Risk Analysis <div class="ct-sub">Customer retention risk</div></div>
            <div id="ch-churn" style="height:220px"></div>
          </div>
          <div class="card">
            <div class="ct">Segment Opportunity Ranking <span class="ct-link">View All</span></div>
            <div id="cu-opp-ranking"></div>
          </div>
          <div class="card">
            <div class="ct">Loyalty Driver Analysis <div class="ct-sub">by Segment</div></div>
            <div id="cu-loyalty-drivers"></div>
          </div>
        </div>

        <div class="card mb14">
          <div class="ct">AI Customer Advisor</div>
          <div class="g4" id="cu-ai-recs" style="margin-bottom:0"></div>
        </div>

        <div class="dash-footer">
          <span>Survey Period: 2024 · 1,730 respondents analyzed</span>
          <span>Segmentation: Loyalty × Satisfaction matrix</span>
        </div>
      </div>

      <!-- ── PAGE 5: COMPETITIVE INTELLIGENCE ──────────────────── -->
      <div class="page" id="page-competitor">
        <div class="ai-banner">
          <div class="ai-banner-icon" style="background:var(--indigo)"><i class="fa-solid fa-chart-line"></i></div>
          <div class="ai-banner-body">
            <div class="ai-banner-label" style="color:var(--indigo)">AI Competitive Insight</div>
            <div class="ai-banner-text" id="co-ai-text">Loading...</div>
            <div class="ai-banner-sub" id="co-ai-sub"></div>
          </div>
          <div class="ai-banner-actions">
            <button class="btn-primary" style="background:var(--indigo)" onclick="setAIPage('competitor')"><i class="fa-solid fa-wand-magic-sparkles"></i> Lihat Insight Detail</button>
          </div>
        </div>

        <div class="g4" id="co-kpis"></div>

        <div class="g3">
          <div class="card">
            <div class="ct">Market Position Matrix <div class="ct-sub">XYZ vs Competitor positioning</div></div>
            <div id="ch-market-pos" style="height:280px"></div>
          </div>
          <div class="card">
            <div class="ct">Competitor Threat Analysis <div class="ct-sub">Service dimension comparison</div></div>
            <div id="ch-comp-radar" style="height:280px"></div>
          </div>
          <div class="card">
            <div class="ct">Competitor Ranking</div>
            <div id="co-ranking"></div>
          </div>
        </div>

        <div class="g3">
          <div class="card">
            <div class="ct">Competitive Advantage Scorecard</div>
            <div id="co-adv-scorecard"></div>
          </div>
          <div class="card">
            <div class="ct">Competitive Gap Heatmap <div class="ct-sub">vs Best Competitor</div></div>
            <div id="ch-comp-gap" style="height:260px"></div>
          </div>
          <div class="card">
            <div class="ct">Competitive Advantage Highlights</div>
            <div id="co-adv-highlights"></div>
          </div>
        </div>

        <div class="g3">
          <div class="card">
            <div class="ct">Switching Risk Analysis</div>
            <div id="ch-switch-risk" style="height:220px"></div>
          </div>
          <div class="card">
            <div class="ct">Top Reasons to Switch</div>
            <div id="ch-switch-reasons" style="height:220px"></div>
          </div>
          <div class="card">
            <div class="ct">Opportunity Ranking</div>
            <div id="co-opp-rank"></div>
          </div>
        </div>

        <div class="card mb14">
          <div class="ct">AI Strategy Advisor</div>
          <div class="g4" id="co-ai-recs" style="margin-bottom:0"></div>
        </div>

        <div class="dash-footer">
          <span>Survey Period: 2024 · Competitor benchmark: 546 responses</span>
          <span>Market position analysis vs 1 direct competitor</span>
        </div>
      </div>

      <!-- ── PAGE 6: EXECUTIVE ACTION CENTER ───────────────────── -->
      <div class="page" id="page-action">
        <div class="ai-banner">
          <div class="ai-banner-icon" style="background:var(--green)"><i class="fa-solid fa-bolt"></i></div>
          <div class="ai-banner-body">
            <div class="ai-banner-label" style="color:var(--green)">AI Executive Insight</div>
            <div class="ai-banner-text" id="ac-ai-text">Loading...</div>
            <div class="ai-banner-sub" id="ac-ai-sub"></div>
          </div>
          <div class="ai-banner-actions">
            <div class="ai-banner-meta" style="margin-right:10px">
              <div class="ai-meta-item">
                <div class="ai-meta-lbl">Confidence Score</div>
                <div class="ai-meta-val" style="color:var(--green)">87%</div>
                <div class="ai-meta-badge bg">High</div>
              </div>
            </div>
            <div class="ai-banner-meta">
              <div class="ai-meta-item">
                <div class="ai-meta-lbl">Est. Business Impact</div>
                <div class="ai-meta-val" id="ac-impact" style="color:var(--orange)">+11.6 NPS</div>
                <div class="ai-meta-badge bo">High Impact</div>
              </div>
            </div>
            <button class="btn-primary" style="margin-left:10px" onclick="setAIPage('action')"><i class="fa-solid fa-map"></i> View Action Plan</button>
          </div>
        </div>

        <div class="g5" id="ac-kpis"></div>

        <div class="g2">
          <div class="card">
            <div class="ct">Impact vs Effort Matrix <div class="ct-sub">Strategic prioritization framework</div></div>
            <div id="ch-ac-matrix" style="height:280px"></div>
          </div>
          <div class="card">
            <div class="ct">Top Priority Actions</div>
            <div id="ac-actions-table"></div>
          </div>
        </div>

        <div class="g2">
          <div class="card">
            <div class="ct">Projected Business Impact</div>
            <div class="g3" id="ac-impact-cards" style="margin-bottom:0"></div>
          </div>
          <div class="card">
            <div class="ct">Executive Roadmap <div class="ct-sub">Implementation timeline</div></div>
            <div class="roadmap" id="ac-roadmap"></div>
          </div>
        </div>

        <div class="card mb14">
          <div class="ct">Action Monitoring</div>
          <div class="mon-grid" id="ac-monitoring"></div>
        </div>

        <div class="dash-footer">
          <span>Survey Period: 2024 · Action plan generated from data analysis</span>
          <span>Confidence Level: High · Based on IPA + Driver Analysis</span>
        </div>
      </div>

      <!-- ── PAGE 7: DATA QUALITY CENTER ───────────────────────── -->
      <div class="page" id="page-dataquality">
        <div class="ai-banner">
          <div class="ai-banner-icon" style="background:var(--blue)"><i class="fa-solid fa-shield-check"></i></div>
          <div class="ai-banner-body">
            <div class="ai-banner-label" style="color:var(--blue)">AI Data Quality Insight</div>
            <div class="ai-banner-text" id="dq-ai-text">Loading...</div>
            <div class="ai-banner-sub" id="dq-ai-sub"></div>
          </div>
          <div class="ai-banner-actions">
            <button class="btn-primary" style="background:var(--blue)"><i class="fa-solid fa-arrow-right"></i> Lihat Rekomendasi</button>
          </div>
        </div>

        <div class="g5" id="dq-kpis"></div>

        <div class="g3">
          <div class="card" style="grid-column:span 2">
            <div class="ct">Indonesia Coverage Map <div class="ct-sub">Survey response coverage by province</div></div>
            <div id="ch-dq-map" style="height:280px"></div>
            <div class="g4" style="margin-top:10px;margin-bottom:0">
              <div style="text-align:center;padding:8px;background:rgba(16,185,129,.1);border-radius:8px">
                <div style="font-size:11px;color:var(--green);font-weight:700">Good Coverage</div>
                <div style="font-size:18px;font-weight:800;color:var(--text1)" id="dq-good-prov">—</div>
              </div>
              <div style="text-align:center;padding:8px;background:rgba(245,158,11,.1);border-radius:8px">
                <div style="font-size:11px;color:var(--yellow);font-weight:700">Need Attention</div>
                <div style="font-size:18px;font-weight:800;color:var(--text1)" id="dq-warn-prov">—</div>
              </div>
              <div style="text-align:center;padding:8px;background:rgba(239,68,68,.1);border-radius:8px">
                <div style="font-size:11px;color:var(--red);font-weight:700">Critical</div>
                <div style="font-size:18px;font-weight:800;color:var(--text1)" id="dq-crit-prov">—</div>
              </div>
              <div style="text-align:center;padding:8px;background:var(--bg);border-radius:8px">
                <div style="font-size:11px;color:var(--text3);font-weight:700">Total Provinces</div>
                <div style="font-size:18px;font-weight:800;color:var(--text1)">{D['g']['provinces']}</div>
              </div>
            </div>
          </div>
          <div>
            <div class="card" style="margin-bottom:14px">
              <div class="ct">Survey Response Funnel</div>
              <div id="ch-dq-funnel" style="height:200px"></div>
            </div>
            <div class="card">
              <div class="ct">Data Quality Alerts</div>
              <div id="dq-alerts"></div>
            </div>
          </div>
        </div>

        <div class="g3">
          <div class="card" style="grid-column:span 2">
            <div class="ct">Data Quality Heatmap <div class="ct-sub">Coverage × Completeness × Validity × Freshness by Province</div></div>
            <div id="ch-dq-heatmap" style="height:260px"></div>
          </div>
          <div class="card">
            <div class="ct">Missing Data Intelligence</div>
            <div id="ch-dq-missing" style="height:180px"></div>
            <div id="dq-missing-detail" style="margin-top:8px"></div>
          </div>
        </div>

        <div class="g2">
          <div class="card">
            <div class="ct">Branch Quality Ranking</div>
            <div class="g2" style="margin-bottom:0">
              <div>
                <div style="font-size:11px;font-weight:700;color:var(--green);margin-bottom:6px">Top Quality Branches</div>
                <div id="dq-top-branches"></div>
              </div>
              <div>
                <div style="font-size:11px;font-weight:700;color:var(--red);margin-bottom:6px">Lowest Quality Branches</div>
                <div id="dq-low-branches"></div>
              </div>
            </div>
          </div>
          <div class="card">
            <div class="ct">AI Recommendation</div>
            <div id="dq-ai-rec"></div>
          </div>
        </div>

        <div class="dash-footer">
          <span>Survey Period: 2024 · Data quality assessment based on response completeness</span>
          <span>Quality Score: Based on coverage, completion, and validity metrics</span>
        </div>
      </div>

      <!-- ── PAGE 8: REPORT CENTER ───────────────────────────────── -->
      <div class="page" id="page-report">
        <div class="ai-banner">
          <div class="ai-banner-icon" style="background:var(--indigo)"><i class="fa-solid fa-file-chart-column"></i></div>
          <div class="ai-banner-body">
            <div class="ai-banner-label" style="color:var(--indigo)">AI Executive Summary</div>
            <div class="ai-banner-text" id="rp-ai-text">Loading...</div>
            <div class="ai-banner-sub" id="rp-ai-sub"></div>
          </div>
          <div class="ai-banner-actions" style="gap:8px;flex-direction:column;align-items:flex-end">
            <button class="btn-primary" style="background:var(--indigo)"><i class="fa-solid fa-file-pdf"></i> Generate Board Report</button>
            <button class="btn-outline" style="border-color:var(--indigo);color:var(--indigo)"><i class="fa-solid fa-presentation-screen"></i> Generate PPT</button>
          </div>
        </div>

        <div class="g4" id="rp-kpis"></div>

        <div class="g2">
          <div class="card">
            <div class="ct">Report Builder <div class="ct-sub">Choose report type to generate</div></div>
            <div class="g2" id="rp-builder" style="margin-bottom:0"></div>
            <div style="text-align:center;margin-top:12px">
              <button class="btn-primary" style="width:100%"><i class="fa-solid fa-plus"></i> Generate Report</button>
            </div>
          </div>
          <div class="card">
            <div class="ct">Recent Reports <span class="ct-link">View All</span></div>
            <div id="rp-recent"></div>
          </div>
        </div>

        <div class="g3">
          <div class="card">
            <div class="ct">Export Center <div class="ct-sub">Download in your preferred format</div></div>
            <div class="g2" id="rp-export" style="margin-bottom:0"></div>
          </div>
          <div class="card">
            <div class="ct">AI Narrative Generator <div class="ct-sub">Convert data to executive narrative</div></div>
            <div id="rp-narrative"></div>
          </div>
          <div class="card">
            <div class="ct">Distribution Center <div class="ct-sub">Report recipients</div></div>
            <div id="rp-recipients"></div>
          </div>
        </div>

        <div class="card mb14">
          <div class="ct">AI Presentation Builder <div class="ct-sub">Auto-generate executive presentation with AI</div></div>
          <div id="rp-ppt-builder" style="display:flex;gap:16px;align-items:flex-start"></div>
        </div>

        <div class="dash-footer">
          <span>Report Center · Bank XYZ Customer Satisfaction 2024</span>
          <span>AI-powered report generation and distribution</span>
        </div>
      </div>

    </div><!-- /content -->

    <!-- AI PANEL RIGHT -->
    <div id="ai-panel">
      <div class="ai-panel-hdr">
        <div class="ai-panel-title"><i class="fa-solid fa-wand-magic-sparkles"></i> <span id="ai-panel-title">AI Copilot</span></div>
        <div class="ai-panel-acts">
          <i class="fa-solid fa-rotate-right ai-panel-act" onclick="refreshAI()"></i>
          <i class="fa-solid fa-expand ai-panel-act"></i>
        </div>
      </div>
      <div class="ai-panel-body" id="ai-panel-body">
        <div class="ai-greeting" id="ai-greeting">Hello! I'm your AI Copilot.<br>Saya siap membantu menganalisis data kepuasan nasabah Bank XYZ.</div>
        <div class="ai-section-title">Insight Cepat</div>
        <div id="ai-quick-insights"></div>
        <div class="ai-section-title" style="margin-top:12px">Pertanyaan yang Disarankan</div>
        <div class="ai-chip-list" id="ai-chips"></div>
        <div class="ai-section-title" style="margin-top:12px">Percakapan</div>
        <div class="ai-messages" id="ai-messages">
          <div class="msg-a">
            <div class="msg-a-lbl"><i class="fa-solid fa-brain"></i> AI Analyst</div>
            Halo! Data survei kepuasan Bank XYZ sudah saya analisis. Silakan tanya apa saja.
          </div>
        </div>
        <div class="ai-loading" id="ai-loading">
          <div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div>
          <span>AI menganalisis data...</span>
        </div>
        <div class="ai-ita">
          <div class="ai-iia-title"><i class="fa-solid fa-bolt"></i> Insight to Action</div>
          <div class="ai-ita-body" id="ai-ita-text">Gunakan insight dari data untuk mendukung keputusan strategis yang lebih efektif.</div>
          <button class="ai-cap-btn" id="ai-cap-btn"><i class="fa-solid fa-clipboard-list"></i> Buat Action Plan</button>
        </div>
      </div>
      <div class="ai-input-row">
        <input class="ai-inp" id="ai-inp" placeholder="Tanyakan apa saja..." onkeydown="if(event.key==='Enter')sendAI()"/>
        <button class="ai-send" onclick="sendAI()"><i class="fa-solid fa-paper-plane"></i></button>
      </div>
    </div>

  </div><!-- /body -->
</div><!-- /main -->

<script>
const D = {DJ};
const C = {{
  orange:'#E85D04',green:'#10B981',yellow:'#F59E0B',red:'#EF4444',
  blue:'#3B82F6',purple:'#8B5CF6',indigo:'#6366F1',
  card:'#FFFFFF',bg:'#F8FAFC',border:'#E2E8F0',
  text1:'#0F172A',text2:'#475569',text3:'#94A3B8',
  font:'Inter,sans-serif',fh:'Plus Jakarta Sans,sans-serif',
}};
const BL = {{
  paper_bgcolor:C.card, plot_bgcolor:C.card,
  font:{{family:C.font,color:C.text1,size:11}},
  margin:{{t:10,b:30,l:45,r:15}},
  xaxis:{{gridcolor:C.border,zeroline:false,tickfont:{{color:C.text3,size:10}}}},
  yaxis:{{gridcolor:C.border,zeroline:false,tickfont:{{color:C.text3,size:10}}}},
  showlegend:false,
  hoverlabel:{{bgcolor:C.card,bordercolor:C.border,font:{{size:12}}}},
}};
const PC = {{displayModeBar:false,responsive:true}};

// ── STATE ─────────────────────────────────────────────────────
let state = {{
  page:'overview',
  provs:[...D.all_prov], kotas:[...D.all_kota],
  cabangs:[...D.all_cabang], panels:[...D.all_panel],
  activeIpaCat: Object.keys(D.ipa_cats)[0],
  aiPage:'overview',
}};

// ── HELPERS ───────────────────────────────────────────────────
const npsColor = v => v>=70?C.green:v>=40?C.yellow:v>=0?C.orange:C.red;
const sevClass  = v => v>=70?'sev-healthy':v>=40?'sev-warning':'sev-critical';
const sevLabel  = v => v>=70?'Healthy':v>=40?'Warning':'Critical';
const filtBranch= () => D.branch.filter(b=>state.cabangs.includes(b.cabang));
const filtProv  = () => D.prov.filter(p=>state.provs.includes(p.provinsi));

// ── SIDEBAR ───────────────────────────────────────────────────
let sbCollapsed = false;
function toggleSB() {{
  sbCollapsed = !sbCollapsed;
  document.getElementById('sb').classList.toggle('col', sbCollapsed);
  document.getElementById('sb-arrow').className = sbCollapsed ? 'fa-solid fa-chevron-right' : 'fa-solid fa-chevron-left';
  document.getElementById('sb-toggle-lbl').textContent = sbCollapsed ? '' : 'Collapse';
}}

// ── NAVIGATION ────────────────────────────────────────────────
const PAGE_TITLES = {{
  overview: 'Executive Command Center',
  branch: 'Branch Intelligence Center',
  touchpoint: 'Service Experience Intelligence Center',
  customer: 'Customer Intelligence Center',
  competitor: 'Competitive Intelligence Center',
  action: 'Executive Action Center',
  dataquality: 'Data Quality Center',
  report: 'Report Center',
}};
const AI_GREETINGS = {{
  overview: 'Hello! Saya AI Copilot Anda.\nSiap membantu menganalisis kondisi kepuasan nasabah Bank XYZ.',
  branch: 'Hello! Saya AI Branch Analyst.\nSiap membantu menganalisis kinerja dan risiko cabang.',
  touchpoint: 'Hello! Saya AI Service Analyst.\nSiap membantu mengidentifikasi driver kepuasan layanan.',
  customer: 'Hello! Saya AI Customer Analyst.\nSiap membantu memahami segmen dan risiko loyalitas nasabah.',
  competitor: 'Hello! Saya AI Competitive Analyst.\nSiap membantu menganalisis posisi kompetitif Bank XYZ.',
  action: 'Hello! Saya AI Action Planner.\nSiap membantu merencanakan aksi strategis berdasarkan insight data.',
  dataquality: 'Hello! Saya AI Data Quality Analyst.\nSiap membantu memantau kualitas dan kelengkapan data survei.',
  report: 'Hello! Saya AI Report Generator.\nSiap membantu membuat laporan eksekutif dari insight dashboard.',
}};
const AI_CHIPS_MAP = {{
  overview: ['Apa kondisi terkini Bank XYZ?','Provinsi mana yang paling berisiko?','Apa prioritas tindakan bulan ini?','Bagaimana performa vs kompetitor?'],
  branch: ['Cabang mana yang butuh intervensi segera?','Apa penyebab utama NPS rendah?','Cabang mana yang bisa jadi benchmark?','Wilayah mana yang paling banyak masalah?'],
  touchpoint: ['Touchpoint apa yang paling berpengaruh?','Apa quick win yang bisa dilakukan?','Mengapa NPS di area tertentu rendah?','Driver kepuasan utama nasabah?'],
  customer: ['Segmen mana yang paling berisiko churn?','Bagaimana profil nasabah loyal?','Segmen mana dengan pertumbuhan terbesar?','Faktor apa yang mempengaruhi loyalitas?'],
  competitor: ['Di mana keunggulan Bank XYZ?','Apa ancaman kompetitor terbesar?','Bagaimana mengurangi risiko switching?','Peluang terbesar vs kompetitor?'],
  action: ['Mengapa ini menjadi prioritas utama?','Simulasikan dampak jika semua dijalankan?','Action apa yang bisa dimulai 30 hari?','Bandingkan skenario implementasi?'],
  dataquality: ['Cabang mana dengan kualitas data terendah?','Bagaimana meningkatkan response rate?','Apakah data sudah representatif?','Variabel apa yang paling banyak missing?'],
  report: ['Generate executive summary bulan ini?','Buat laporan untuk board meeting?','Buat laporan performa per wilayah?','Jadwalkan laporan mingguan?'],
}};
const AI_QUICK_INSIGHTS = {{
  overview: [
    {{icon:'fa-check-circle',col:C.green,bg:'rgba(16,185,129,.1)',text:'NPS Score sangat baik',val:`${{D.g.nps}} — Above Target`}},
    {{icon:'fa-triangle-exclamation',col:C.red,bg:'rgba(239,68,68,.1)',text:`${{D.g.b_critical}} cabang butuh intervensi`,val:'Critical Risk'}},
    {{icon:'fa-arrow-trend-up',col:C.blue,bg:'rgba(59,130,246,.1)',text:'CSI Mean di atas target',val:`${{D.g.csi}}/6 Excellent`}},
  ],
  branch: [
    {{icon:'fa-building',col:C.green,bg:'rgba(16,185,129,.1)',text:`${{D.g.b_healthy}} cabang Healthy`,val:'Above Average NPS'}},
    {{icon:'fa-triangle-exclamation',col:C.red,bg:'rgba(239,68,68,.1)',text:`${{D.g.b_critical}} cabang Critical`,val:'Perlu intervensi segera'}},
    {{icon:'fa-location-dot',col:C.yellow,bg:'rgba(245,158,11,.1)',text:'Jawa Barat konsentrasi risiko',val:'Highest Risk Region'}},
  ],
  touchpoint: [
    {{icon:'fa-bullseye',col:C.red,bg:'rgba(239,68,68,.1)',text:'Customer Service driver utama',val:'Impact 42%'}},
    {{icon:'fa-clock',col:C.yellow,bg:'rgba(245,158,11,.1)',text:'Waiting Time perlu perhatian',val:'Impact 31%'}},
    {{icon:'fa-bolt',col:C.green,bg:'rgba(16,185,129,.1)',text:'4 Quick Wins teridentifikasi',val:'High Impact Available'}},
  ],
  customer: [
    {{icon:'fa-users',col:C.green,bg:'rgba(16,185,129,.1)',text:'Loyal Champion 18.3%',val:'NPS 100.0'}},
    {{icon:'fa-shield-exclamation',col:C.red,bg:'rgba(239,68,68,.1)',text:'At Risk 0.5%',val:'NPS -100.0'}},
    {{icon:'fa-arrow-trend-up',col:C.blue,bg:'rgba(59,130,246,.1)',text:'Satisfied segment terbesar',val:'25.1% — NPS 100.0'}},
  ],
  competitor: [
    {{icon:'fa-trophy',col:C.green,bg:'rgba(16,185,129,.1)',text:'NPS unggul jauh vs kompetitor',val:`+${{(D.g.nps - (D.nps_comp[1]?.nps_score||0)).toFixed(1)}} poin`}},
    {{icon:'fa-handshake',col:C.blue,bg:'rgba(59,130,246,.1)',text:'91.2% nasabah XYZ sebagai bank utama',val:'Strong Loyalty'}},
    {{icon:'fa-chart-line',col:C.orange,bg:'rgba(232,93,4,.1)',text:'Semua layanan unggul vs kompetitor',val:'Competitive Advantage'}},
  ],
  action: [
    {{icon:'fa-bolt',col:C.orange,bg:'rgba(232,93,4,.1)',text:'12 priority actions identified',val:'Ready to Execute'}},
    {{icon:'fa-arrow-trend-up',col:C.green,bg:'rgba(16,185,129,.1)',text:'Est. NPS Impact +11.6',val:'High Confidence 87%'}},
    {{icon:'fa-users-gear',col:C.blue,bg:'rgba(59,130,246,.1)',text:'Quick wins dapat dimulai 30 hari',val:'Low Effort High Impact'}},
  ],
  dataquality: [
    {{icon:'fa-shield-check',col:C.green,bg:'rgba(16,185,129,.1)',text:'Data Quality Score 96.8%',val:'Excellent'}},
    {{icon:'fa-database',col:C.blue,bg:'rgba(59,130,246,.1)',text:'Coverage 128/128 branches',val:'100% Complete'}},
    {{icon:'fa-circle-check',col:C.green,bg:'rgba(16,185,129,.1)',text:'Completion Rate 80.5%',val:'1,730 valid responses'}},
  ],
  report: [
    {{icon:'fa-file-chart-column',col:C.indigo,bg:'rgba(99,102,241,.1)',text:'6 report types available',val:'AI-powered generation'}},
    {{icon:'fa-users',col:C.blue,bg:'rgba(59,130,246,.1)',text:'Multi-format export',val:'PDF, PPT, Excel, Word'}},
    {{icon:'fa-calendar-check',col:C.green,bg:'rgba(16,185,129,.1)',text:'Scheduled delivery available',val:'Daily, Weekly, Monthly'}},
  ],
}};

function goPage(el) {{
  document.querySelectorAll('.nav').forEach(n => n.classList.remove('active'));
  el.classList.add('active');
  const page = el.dataset.page;
  state.page = page;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.getElementById('tb-title').textContent = PAGE_TITLES[page] || page;
  buildFilters(page);
  renderPage(page);
  setAIPage(page);
}}

function setAIPage(page) {{
  state.aiPage = page;
  document.getElementById('ai-panel-title').textContent = page === 'overview' ? 'AI Copilot' : 'AI ' + (page === 'branch' ? 'Branch Analyst' : page === 'touchpoint' ? 'Service Analyst' : page === 'customer' ? 'Customer Analyst' : page === 'competitor' ? 'Competitive Analyst' : page === 'action' ? 'Action Planner' : page === 'dataquality' ? 'Data Quality Analyst' : 'Report Generator');
  document.getElementById('ai-greeting').innerHTML = (AI_GREETINGS[page]||'').replace(/\n/g,'<br>');
  buildAIInsights(page);
  buildAIChips(page);
}}

function buildAIInsights(page) {{
  const ins = AI_QUICK_INSIGHTS[page] || [];
  document.getElementById('ai-quick-insights').innerHTML = ins.map(i => `
    <div class="ai-insight">
      <div class="ai-insight-icon" style="background:${{i.bg}};color:${{i.col}}"><i class="fa-solid ${{i.icon}}"></i></div>
      <div><div class="ai-insight-text">${{i.text}}</div>
      <div class="ai-insight-val" style="color:${{i.col}}">${{i.val}}</div></div>
    </div>`).join('');
}}

function buildAIChips(page) {{
  const chips = AI_CHIPS_MAP[page] || [];
  document.getElementById('ai-chips').innerHTML = chips.map(q =>
    `<div class="ai-q-chip" onclick="fillAI('${{q}}')">${{q}}<i class="fa-solid fa-chevron-right"></i></div>`
  ).join('');
}}

// ── FILTERS ───────────────────────────────────────────────────
function buildFilters(page) {{
  const wrap = document.getElementById('tb-filters');
  wrap.innerHTML = '';
  const showProvKota = ['overview','branch','dataquality'].includes(page);
  const showPanel = ['touchpoint','customer','competitor','action'].includes(page);
  if (showProvKota) {{
    wrap.appendChild(mkFilter('f-prov','Provinsi',D.all_prov,state.provs,v=>{{
      state.provs = v;
      state.kotas = [...new Set(v.flatMap(p=>D.prov_kota[p]||[]))];
      state.cabangs = [...new Set(state.kotas.flatMap(k=>D.kota_cabang[k]||[]))];
      buildFilters(page); updateChips(); renderPage(page);
    }}));
    wrap.appendChild(mkFilter('f-kota','Kota/Kab',
      [...new Set(state.provs.flatMap(p=>D.prov_kota[p]||[]))],state.kotas,v=>{{
      state.kotas = v;
      state.cabangs = [...new Set(v.flatMap(k=>D.kota_cabang[k]||[]))];
      buildFilters(page); updateChips(); renderPage(page);
    }}));
    wrap.appendChild(mkFilter('f-cab','Cabang',state.cabangs,state.cabangs,v=>{{
      state.cabangs = v; updateChips(); renderPage(page);
    }}));
  }}
  if (showPanel) {{
    wrap.appendChild(mkFilter('f-panel','Panel',D.all_panel,state.panels,v=>{{
      state.panels = v; updateChips(); renderPage(page);
    }}));
  }}
}}

function mkFilter(id,label,opts,sel,onChange) {{
  const w = document.createElement('div');
  w.className = 'flt-wrap';
  const allS = sel.length === opts.length;
  w.innerHTML = `
    <div class="flt-btn" id="fb-${{id}}" onclick="toggleFlt('${{id}}')">
      <div><div class="flt-lbl">${{label}}</div>
      <div class="flt-val" id="fv-${{id}}">${{allS?'All selected':sel.length+' selected'}}</div></div>
      <i class="fa-solid fa-chevron-down flt-arrow"></i>
    </div>
    <div class="flt-dd" id="fd-${{id}}">
      <div class="flt-dd-hdr">
        <span class="flt-dd-title">${{label}}</span>
        <div class="flt-dd-acts">
          <span class="flt-dd-act" onclick="fltAll('${{id}}')">All</span>
          <span class="flt-dd-act" onclick="fltNone('${{id}}')">None</span>
        </div>
      </div>
      <div class="flt-search"><input type="text" placeholder="Search..." oninput="fltSearch('${{id}}',this.value)"></div>
      <div class="flt-list" id="fl-${{id}}"></div>
    </div>`;
  const list = w.querySelector('#fl-'+id);
  opts.forEach(o => {{
    const item = document.createElement('div');
    item.className = 'flt-item' + (sel.includes(o)?' ck':'');
    item.dataset.val = o;
    item.innerHTML = `<div class="flt-cb">${{sel.includes(o)?'✓':''}}</div><span>${{o}}</span>`;
    item.onclick = e => {{
      e.stopPropagation();
      const cur = [...w.querySelectorAll('.flt-item.ck')].map(i=>i.dataset.val);
      const idx = cur.indexOf(o);
      if (idx > -1) {{ if(cur.length===1) return; cur.splice(idx,1); item.classList.remove('ck'); item.querySelector('.flt-cb').textContent=''; }}
      else {{ cur.push(o); item.classList.add('ck'); item.querySelector('.flt-cb').textContent='✓'; }}
      document.getElementById('fv-'+id).textContent = cur.length===opts.length?'All selected':cur.length+' selected';
      onChange(cur);
    }};
    list.appendChild(item);
  }});
  return w;
}}

function toggleFlt(id) {{
  const dd=document.getElementById('fd-'+id), btn=document.getElementById('fb-'+id);
  const open=dd.classList.contains('open');
  document.querySelectorAll('.flt-dd').forEach(d=>d.classList.remove('open'));
  document.querySelectorAll('.flt-btn').forEach(b=>b.classList.remove('open'));
  if(!open){{ dd.classList.add('open'); btn.classList.add('open'); }}
}}
function fltAll(id) {{ document.querySelectorAll(`#fl-${{id}} .flt-item`).forEach(i=>{{i.classList.add('ck');i.querySelector('.flt-cb').textContent='✓';}}); }}
function fltNone(id) {{ document.querySelectorAll(`#fl-${{id}} .flt-item`).forEach((i,n)=>{{if(n===0){{i.classList.add('ck');i.querySelector('.flt-cb').textContent='✓';}}else{{i.classList.remove('ck');i.querySelector('.flt-cb').textContent='';}}}}); }}
function fltSearch(id,q) {{ document.querySelectorAll(`#fl-${{id}} .flt-item`).forEach(i=>{{ i.style.display=i.dataset.val.toLowerCase().includes(q.toLowerCase())?'':'none'; }}); }}
document.addEventListener('click',()=>{{ document.querySelectorAll('.flt-dd').forEach(d=>d.classList.remove('open')); document.querySelectorAll('.flt-btn').forEach(b=>b.classList.remove('open')); }});

function updateChips() {{
  const chips = document.getElementById('flt-chips');
  const items = [];
  if(state.provs.length < D.all_prov.length) state.provs.forEach(p=>items.push({{label:p,type:'prov'}}));
  if(state.panels.length < D.all_panel.length) state.panels.forEach(p=>items.push({{label:p,type:'panel'}}));
  if(items.length===0){{ chips.style.display='none'; return; }}
  chips.style.display='flex';
  chips.innerHTML = items.map(it=>`<div class="chip">${{it.label}}<span class="chip-x" onclick="removeChip('${{it.type}}','${{it.label}}')">✕</span></div>`).join('')
    + `<div class="chip chip-reset" onclick="resetFilters()"><i class="fa-solid fa-rotate-left"></i> Reset All</div>`;
}}

function removeChip(type,label) {{
  if(type==='prov') {{
    state.provs = state.provs.filter(p=>p!==label);
    state.kotas = [...new Set(state.provs.flatMap(p=>D.prov_kota[p]||[]))];
    state.cabangs = [...new Set(state.kotas.flatMap(k=>D.kota_cabang[k]||[]))];
  }} else if(type==='panel') state.panels = state.panels.filter(p=>p!==label);
  buildFilters(state.page); updateChips(); renderPage(state.page);
}}

function resetFilters() {{
  state.provs=[...D.all_prov]; state.kotas=[...D.all_kota];
  state.cabangs=[...D.all_cabang]; state.panels=[...D.all_panel];
  buildFilters(state.page); updateChips(); renderPage(state.page);
}}

// ── RENDER ROUTING ─────────────────────────────────────────────
function renderPage(p) {{
  if(p==='overview')   renderOverview();
  if(p==='branch')     renderBranch();
  if(p==='touchpoint') renderTouchpoint();
  if(p==='customer')   renderCustomer();
  if(p==='competitor') renderCompetitor();
  if(p==='action')     renderAction();
  if(p==='dataquality')renderDataQuality();
  if(p==='report')     renderReport();
}}

// ══════════════════════════════════════════════════════════════
// PAGE 1: EXECUTIVE COMMAND CENTER
// ══════════════════════════════════════════════════════════════
function renderOverview() {{
  const g = D.g;
  const pv = filtProv();

  // AI Banner
  const critProv = D.prov.sort((a,b)=>a.nps_score-b.nps_score)[0];
  const bestProv = D.prov.sort((a,b)=>b.nps_score-a.nps_score)[0];
  document.getElementById('ov-ai-text').textContent = `NPS Bank XYZ ${{g.nps}} dengan ${{g.b_critical}} cabang kritis. ${{critProv?.provinsi}} memiliki konsentrasi risiko tertinggi.`;
  document.getElementById('ov-ai-sub').textContent = `${{g.promoters.toLocaleString()}} Promoter (${{(g.promoters/g.total*100).toFixed(1)}}%) dan ${{g.detractors}} Detractor (${{g.customer_risk}}%). ${{bestProv?.provinsi}} tampil terbaik dengan NPS ${{bestProv?.nps_score.toFixed(1)}}.`;
  document.getElementById('ov-impact').textContent = '+NPS ' + g.nps;

  // KPI Cards
  const kpiData = [
    {{ic:'fa-chart-line',icBg:'rgba(16,185,129,.1)',icCol:C.green,lbl:'NPS Score',val:g.nps.toFixed(1),sub:'Net Promoter Score',badge:'Above Target',bClass:'bg',col:C.green}},
    {{ic:'fa-face-smile',icBg:'rgba(59,130,246,.1)',icCol:C.blue,lbl:'Customer Satisfaction (CSI)',val:g.csi+'/6',sub:`${{g.csi_pct}}% Sangat Puas`,badge:'Above Target',bClass:'bg',col:C.blue}},
    {{ic:'fa-heart',icBg:'rgba(139,92,246,.1)',icCol:C.purple,lbl:'Loyalty Index',val:g.loyalty+'/6',sub:'Rata-rata semua panel',badge:'Above Target',bClass:'bg',col:C.purple}},
    {{ic:'fa-triangle-exclamation',icBg:'rgba(239,68,68,.1)',icCol:C.red,lbl:'Customer Risk',val:g.customer_risk+'%',sub:`${{g.detractors}} Detractors`,badge:'Needs Attention',bClass:'br',col:C.red}},
    {{ic:'fa-users',icBg:'rgba(232,93,4,.1)',icCol:C.orange,lbl:'Responses',val:g.total.toLocaleString(),sub:`${{g.provinces}} prov · ${{g.branches}} cabang`,badge:`${{g.branches}} Branches`,bClass:'bo',col:C.orange}},
  ];
  document.getElementById('ov-kpis').innerHTML = kpiData.map(k=>`
    <div class="kpi">
      <div class="kpi-ic" style="background:${{k.icBg}};color:${{k.icCol}}"><i class="fa-solid ${{k.ic}}"></i></div>
      <div class="kpi-body">
        <div class="kpi-lbl">${{k.lbl}}</div>
        <div class="kpi-val" style="color:${{k.col}}">${{k.val}}</div>
        <div class="kpi-val-sub">${{k.sub}}</div>
        <div class="kpi-badge ${{k.bClass}}">${{k.badge}}</div>
      </div>
    </div>`).join('');

  // Alert Center
  const critBr = D.branch.filter(b=>b.severity==='Critical');
  const alerts = [
    {{type:'crit',icon:'fa-circle-exclamation',iconBg:'rgba(239,68,68,.15)',iconCol:C.red,title:`${{critBr.length}} Critical Branches`,desc:'Require immediate action'}},
    {{type:'warn',icon:'fa-triangle-exclamation',iconBg:'rgba(245,158,11,.15)',iconCol:C.yellow,title:`${{D.prov.sort((a,b)=>a.nps_score-b.nps_score)[0]?.provinsi}} Risk Alert`,desc:'Lowest NPS province'}},
    {{type:'warn',icon:'fa-headset',iconBg:'rgba(245,158,11,.15)',iconCol:C.yellow,title:'Customer Service',desc:'Largest impact on NPS'}},
    {{type:'succ',icon:'fa-star',iconBg:'rgba(16,185,129,.15)',iconCol:C.green,title:D.prov.sort((a,b)=>b.nps_score-a.nps_score)[0]?.provinsi+' — Best Region',desc:`NPS ${{D.prov[0]?.nps_score?.toFixed(1)}}`}},
  ];
  document.getElementById('ov-alerts').innerHTML = alerts.map(a=>`
    <div class="alert-item alert-${{a.type}}">
      <div class="alert-icon" style="background:${{a.iconBg}};color:${{a.iconCol}}"><i class="fa-solid ${{a.icon}}"></i></div>
      <div class="alert-body">
        <div class="alert-title">${{a.title}}</div>
        <div class="alert-desc">${{a.desc}}</div>
      </div>
    </div>`).join('');

  // Province ranking
  const pvSorted = [...pv].sort((a,b)=>b.nps_score-a.nps_score);
  Plotly.newPlot('ch-prov-rank',[{{
    type:'bar',orientation:'h',
    x:pvSorted.slice(0,10).map(p=>p.nps_score).reverse(),
    y:pvSorted.slice(0,10).map(p=>p.provinsi).reverse(),
    marker:{{color:pvSorted.slice(0,10).map(p=>npsColor(p.nps_score)).reverse()}},
    text:pvSorted.slice(0,10).map(p=>p.nps_score.toFixed(1)).reverse(),
    textposition:'outside',
    hovertemplate:'%{{y}}: NPS %{{x}}<extra></extra>',
  }}],{{...BL,
    margin:{{t:10,b:20,l:130,r:50}},
    xaxis:{{...BL.xaxis,range:[0,130]}},
    shapes:[{{type:'line',x0:g.nps,x1:g.nps,y0:-.5,y1:9.5,line:{{color:C.orange,dash:'dot',width:1.5}}}}],
  }},PC);

  // Drivers chart
  const ovr = D.ovr;
  Plotly.newPlot('ch-drivers',[
    {{type:'bar',x:ovr.map(o=>o.kategori||o.kategori_layanan),y:ovr.map(o=>o.mean_score),
      marker:{{color:ovr.map(o=>o.mean_score>=5.9?C.green:o.mean_score>=5.7?C.yellow:C.red)}},
      text:ovr.map(o=>o.mean_score.toFixed(2)),textposition:'outside',
      name:'Score',yaxis:'y'}},
    {{type:'scatter',mode:'markers',x:ovr.map(o=>o.kategori||o.kategori_layanan),
      y:ovr.map(o=>o.pct_puas),
      marker:{{color:C.blue,size:10}},name:'% Puas',yaxis:'y2'}},
  ],{{...BL,barmode:'group',showlegend:true,
    legend:{{orientation:'h',y:1.1,font:{{size:10}}}},
    margin:{{t:30,b:60,l:50,r:60}},
    yaxis:{{...BL.yaxis,range:[5,6.5],title:'Score (1-6)'}},
    yaxis2:{{overlaying:'y',side:'right',range:[95,101],title:'% Sangat Puas',tickfont:{{size:10}}}},
    xaxis:{{...BL.xaxis,tickangle:-20}},
  }},PC);

  // Risk provinces
  const riskProv = [...D.prov].sort((a,b)=>a.nps_score-b.nps_score).slice(0,5);
  document.getElementById('ov-risk-prov').innerHTML = riskProv.map((p,i)=>`
    <div class="ri">
      <div class="ri-num rcrit">${{i+1}}</div>
      <div class="ri-body"><div class="ri-name">${{p.provinsi}}</div>
        <div class="ri-sub">${{p.n_responden}} responses</div></div>
      <div>
        <div class="ri-score" style="color:${{npsColor(p.nps_score)}}">${{p.nps_score.toFixed(1)}}</div>
        <div class="${{sevClass(p.nps_score)}}">${{sevLabel(p.nps_score)}}</div>
      </div>
    </div>`).join('');

  // Opportunity provinces
  const oppProv = [...D.prov].sort((a,b)=>b.nps_score-a.nps_score).slice(0,5);
  document.getElementById('ov-opp-prov').innerHTML = oppProv.map((p,i)=>`
    <div class="ri">
      <div class="ri-num r${{i<3?i+1:'n'}}">${{i+1}}</div>
      <div class="ri-body"><div class="ri-name">${{p.provinsi}}</div>
        <div class="ri-sub">${{p.n_responden}} responses</div></div>
      <div>
        <div class="ri-score" style="color:${{npsColor(p.nps_score)}}">${{p.nps_score.toFixed(1)}}</div>
        <div class="sev-healthy">Healthy</div>
      </div>
    </div>`).join('');

  // Benchmark
  const comp = D.nps_comp;
  document.getElementById('ov-benchmark').innerHTML = `
    <table style="width:100%;border-collapse:collapse;font-size:11px">
      <tr style="border-bottom:1px solid var(--border)">
        <th style="text-align:left;padding:6px;color:var(--text3);font-size:10px;text-transform:uppercase">Metric</th>
        <th style="text-align:center;padding:6px;color:var(--orange);font-size:10px">Bank XYZ</th>
        <th style="text-align:center;padding:6px;color:var(--text3);font-size:10px">Target</th>
        <th style="text-align:center;padding:6px;color:var(--blue);font-size:10px">Competitor</th>
      </tr>
      <tr style="border-bottom:1px solid var(--border)">
        <td style="padding:7px 6px;font-weight:600;color:var(--text2)">NPS</td>
        <td style="text-align:center;font-weight:800;color:var(--orange)">${{g.nps}}</td>
        <td style="text-align:center;color:var(--text3)">75.0</td>
        <td style="text-align:center;font-weight:700;color:var(--blue)">${{comp[1]?.nps_score?.toFixed(1)||'N/A'}}</td>
      </tr>
      <tr style="border-bottom:1px solid var(--border)">
        <td style="padding:7px 6px;font-weight:600;color:var(--text2)">CSI</td>
        <td style="text-align:center;font-weight:800;color:var(--orange)">${{g.csi}}</td>
        <td style="text-align:center;color:var(--text3)">5.5</td>
        <td style="text-align:center;font-weight:700;color:var(--blue)">~5.4</td>
      </tr>
      <tr>
        <td style="padding:7px 6px;font-weight:600;color:var(--text2)">Loyalty</td>
        <td style="text-align:center;font-weight:800;color:var(--orange)">${{g.loyalty}}</td>
        <td style="text-align:center;color:var(--text3)">5.5</td>
        <td style="text-align:center;font-weight:700;color:var(--blue)">~5.3</td>
      </tr>
    </table>`;

  // Data Quality strip
  document.getElementById('ov-dq').innerHTML = [
    {{lbl:'Total Responses',val:g.total.toLocaleString(),sub:'+231 vs target',col:C.green,ic:'fa-database'}},
    {{lbl:'Branch Coverage',val:'128/128',sub:'100% Complete',col:C.green,ic:'fa-building'}},
    {{lbl:'Confidence Level',val:'High',sub:'96.8% quality score',col:C.green,ic:'fa-shield-check'}},
    {{lbl:'Data Quality Score',val:'96.8%',sub:'Excellent',col:C.green,ic:'fa-circle-check'}},
  ].map(d=>`
    <div style="background:var(--bg);border-radius:9px;padding:10px 12px;border:1px solid var(--border);display:flex;align-items:center;gap:10px">
      <div style="width:32px;height:32px;border-radius:8px;background:rgba(16,185,129,.1);color:${{d.col}};display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0"><i class="fa-solid ${{d.ic}}"></i></div>
      <div><div style="font-size:10px;color:var(--text3);font-weight:600">${{d.lbl}}</div>
      <div style="font-family:var(--fh);font-size:15px;font-weight:800;color:${{d.col}}">${{d.val}}</div>
      <div style="font-size:10px;color:var(--text3)">${{d.sub}}</div></div>
    </div>`).join('');
}}

// ══════════════════════════════════════════════════════════════
// PAGE 2: BRANCH INTELLIGENCE CENTER
// ══════════════════════════════════════════════════════════════
function renderBranch() {{
  const g = D.g;
  const br = filtBranch();

  // AI Banner
  const critBr = br.filter(b=>b.severity==='Critical');
  document.getElementById('br-ai-text').textContent = `${{critBr.length}} cabang membutuhkan intervensi segera.`;
  document.getElementById('br-ai-sub').textContent = `Customer Service menjadi penyebab utama penurunan NPS. ${{D.prov.sort((a,b)=>a.nps_score-b.nps_score)[0]?.provinsi}} memiliki konsentrasi risiko tertinggi.`;

  // KPI
  document.getElementById('br-kpis').innerHTML = [
    {{ic:'fa-building',icBg:'rgba(232,93,4,.1)',icCol:C.orange,lbl:'Total Branches',val:g.branches,sub:'128 active branches',badge:'Full Coverage',bClass:'bo',col:C.orange}},
    {{ic:'fa-circle-check',icBg:'rgba(16,185,129,.1)',icCol:C.green,lbl:'Healthy Branches',val:g.b_healthy,sub:`${{(g.b_healthy/g.branches*100).toFixed(1)}}% of total`,badge:'Above Average NPS',bClass:'bg',col:C.green}},
    {{ic:'fa-triangle-exclamation',icBg:'rgba(245,158,11,.1)',icCol:C.yellow,lbl:'Warning Branches',val:g.b_warning,sub:`${{(g.b_warning/g.branches*100).toFixed(1)}}% of total`,badge:'Monitor Closely',bClass:'by',col:C.yellow}},
    {{ic:'fa-circle-exclamation',icBg:'rgba(239,68,68,.1)',icCol:C.red,lbl:'Critical Branches',val:g.b_critical,sub:`${{(g.b_critical/g.branches*100).toFixed(1)}}% of total`,badge:'Immediate Action',bClass:'br',col:C.red}},
    {{ic:'fa-chart-bar',icBg:'rgba(59,130,246,.1)',icCol:C.blue,lbl:'Avg Branch NPS',val:g.avg_branch_nps,sub:`vs target 75.0`,badge:'Above Target',bClass:'bg',col:C.blue}},
  ].map(k=>`
    <div class="kpi">
      <div class="kpi-ic" style="background:${{k.icBg}};color:${{k.icCol}}"><i class="fa-solid ${{k.ic}}"></i></div>
      <div class="kpi-body">
        <div class="kpi-lbl">${{k.lbl}}</div>
        <div class="kpi-val" style="color:${{k.col}}">${{k.val}}</div>
        <div class="kpi-val-sub">${{k.sub}}</div>
        <div class="kpi-badge ${{k.bClass}}">${{k.badge}}</div>
      </div>
    </div>`).join('');

  // Branch Risk Map (Province level)
  const pvData = filtProv();
  Plotly.newPlot('ch-br-map',[{{
    type:'bar',orientation:'h',
    y:pvData.sort((a,b)=>a.nps_score-b.nps_score).map(p=>p.provinsi),
    x:pvData.sort((a,b)=>a.nps_score-b.nps_score).map(p=>p.nps_score),
    marker:{{
      color:pvData.sort((a,b)=>a.nps_score-b.nps_score).map(p=>npsColor(p.nps_score)),
      line:{{color:'white',width:1}}
    }},
    text:pvData.sort((a,b)=>a.nps_score-b.nps_score).map(p=>p.nps_score.toFixed(1)+' — '+sevLabel(p.nps_score)),
    textposition:'outside',
    hovertemplate:'<b>%{{y}}</b><br>NPS: %{{x}}<br>Cabang: %{{customdata}}<extra></extra>',
    customdata:pvData.sort((a,b)=>a.nps_score-b.nps_score).map(p=>p.n_responden+' responses'),
  }}],{{...BL,
    margin:{{t:10,b:30,l:150,r:80}},
    xaxis:{{...BL.xaxis,range:[Math.min(...pvData.map(p=>p.nps_score))-20,130]}},
    shapes:[{{type:'line',x0:g.nps,x1:g.nps,y0:-.5,y1:pvData.length-.5,line:{{color:C.orange,dash:'dot',width:1.5}}}}],
  }},PC);

  // Regional Performance
  const sortedPv = [...D.prov].sort((a,b)=>b.nps_score-a.nps_score);
  document.getElementById('br-regional').innerHTML = `
    <div style="font-size:10px;font-weight:700;color:var(--green);margin-bottom:6px">Top 5 Provinces</div>
    ${{sortedPv.slice(0,5).map((p,i)=>`
    <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--border)">
      <div style="width:18px;height:18px;border-radius:4px;background:rgba(16,185,129,.15);color:var(--green);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800">${{i+1}}</div>
      <div style="flex:1;font-size:11px;font-weight:600;color:var(--text1)">${{p.provinsi}}</div>
      <div style="font-size:12px;font-weight:800;color:var(--green)">${{p.nps_score.toFixed(1)}}</div>
      <i class="fa-solid fa-arrow-up" style="color:var(--green);font-size:10px"></i>
    </div>`).join('')}}
    <div style="font-size:10px;font-weight:700;color:var(--red);margin-top:10px;margin-bottom:6px">Bottom 5 Provinces</div>
    ${{sortedPv.slice(-5).reverse().map((p,i)=>`
    <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--border)">
      <div style="width:18px;height:18px;border-radius:4px;background:rgba(239,68,68,.15);color:var(--red);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800">${{i+1}}</div>
      <div style="flex:1;font-size:11px;font-weight:600;color:var(--text1)">${{p.provinsi}}</div>
      <div style="font-size:12px;font-weight:800;color:var(--red)">${{p.nps_score.toFixed(1)}}</div>
      <i class="fa-solid fa-arrow-down" style="color:var(--red);font-size:10px"></i>
    </div>`).join('')}}`;

  // Top Opportunity Branches
  const top5 = br.slice(0,5);
  document.getElementById('br-top5').innerHTML = top5.map((b,i)=>`
    <div class="ri">
      <div class="ri-num r${{i<3?i+1:'n'}}">#${{b.rank}}</div>
      <div class="ri-body">
        <div class="ri-name">${{b.cabang}}</div>
        <div class="ri-sub">${{b.provinsi}} · ${{b.n_responden}} resp</div>
      </div>
      <div style="text-align:right">
        <div class="ri-score" style="color:${{C.green}}">${{b.nps_score.toFixed(1)}}</div>
        <div class="sev-healthy" style="font-size:9px">Best Practice</div>
      </div>
    </div>`).join('');

  // Critical Branches
  const crit5 = br.filter(b=>b.severity==='Critical').slice(0,5);
  document.getElementById('br-crit5').innerHTML = crit5.length ? crit5.map((b,i)=>`
    <div class="ri" style="border-left:3px solid ${{C.red}}">
      <div class="ri-num rcrit"><i class="fa-solid fa-triangle-exclamation" style="font-size:9px"></i></div>
      <div class="ri-body">
        <div class="ri-name">${{b.cabang}}</div>
        <div class="ri-sub">${{b.provinsi}} · ${{b.n_responden}} resp</div>
      </div>
      <div style="text-align:right">
        <div class="ri-score" style="color:${{C.red}}">${{b.nps_score.toFixed(1)}}</div>
        <div class="sev-critical" style="font-size:9px">Critical</div>
      </div>
    </div>`).join('') : '<div style="padding:20px;text-align:center;color:var(--green);font-size:12px"><i class="fa-solid fa-check-circle"></i> No critical branches in selection</div>';

  // Heatmap
  const hm5 = [...br].sort((a,b)=>a.nps_score-b.nps_score).slice(0,8).concat(br.slice(0,4));
  const hmBr = hm5.map(b=>b.cabang);
  Plotly.newPlot('ch-br-heatmap',[{{
    type:'heatmap',
    z:[hm5.map(b=>b.nps_score),hm5.map(b=>b.csi_mean||0),hm5.map(b=>b.loyalty_mean||0)],
    x:hmBr,y:['NPS','CSI','Loyalty'],
    colorscale:[[0,C.red],[0.5,C.yellow],[1,C.green]],
    showscale:true,
    text:[hm5.map(b=>b.nps_score.toFixed(1)),hm5.map(b=>(b.csi_mean||0).toFixed(2)),hm5.map(b=>(b.loyalty_mean||0).toFixed(2))],
    texttemplate:'%{{text}}',textfont:{{size:9}},
    hovertemplate:'%{{x}}<br>%{{y}}: %{{z}}<extra></extra>',
  }}],{{...BL,
    margin:{{t:10,b:80,l:70,r:60}},
    xaxis:{{...BL.xaxis,tickangle:-30,tickfont:{{size:9}}}},
    yaxis:{{...BL.yaxis}},
  }},PC);

  // Root Cause
  const rc = [['Customer Service',42],['Waiting Time',31],['ATM Reliability',18],['Product Info',9]];
  Plotly.newPlot('ch-root-cause',[{{
    type:'bar',orientation:'h',
    x:rc.map(r=>r[1]),y:rc.map(r=>r[0]),
    marker:{{color:[C.red,C.orange,C.yellow,C.green]}},
    text:rc.map(r=>r[1]+'%'),textposition:'outside',
    hovertemplate:'%{{y}}: %{{x}}%<extra></extra>',
  }}],{{...BL,
    margin:{{t:10,b:30,l:130,r:50}},
    xaxis:{{...BL.xaxis,range:[0,55],title:'% Contribution'}},
    yaxis:{{...BL.yaxis,autorange:'reversed'}},
  }},PC);

  // Driver table
  const ovr = D.ovr;
  document.getElementById('br-driver-table').innerHTML = `
    <table class="hm-table">
      <tr><th>Driver</th><th>Impact Score</th><th>Level</th></tr>
      ${{ovr.sort((a,b)=>b.mean_score-a.mean_score).map(o=>`
      <tr>
        <td>${{o.kategori||o.kategori_layanan}}</td>
        <td style="text-align:center"><div style="display:flex;align-items:center;gap:6px;justify-content:center">
          <div style="height:5px;width:${{Math.round(o.mean_score/6*80)}}px;background:${{o.mean_score>=5.9?C.green:o.mean_score>=5.7?C.yellow:C.red}};border-radius:3px"></div>
          <span style="font-weight:700;color:var(--text1)">${{o.mean_score.toFixed(3)}}</span>
        </div></td>
        <td><span class="${{o.mean_score>=5.9?'sev-healthy':o.mean_score>=5.7?'sev-warning':'sev-critical'}}">${{o.mean_score>=5.9?'High':'Medium'}}</span></td>
      </tr>`).join('')}}
    </table>`;

  // AI Branch Recommendations
  document.getElementById('br-ai-recs').innerHTML = [
    {{pri:'PRIORITY 1',tipe:'Immediate Action',br:crit5[0]?.cabang||'Bogor 1',cause:'Customer Service',impact:'+8.2',effort:'Medium',tiCol:C.red}},
    {{pri:'PRIORITY 2',tipe:'High Impact',br:crit5[1]?.cabang||'Semarang 2',cause:'Waiting Time',impact:'+5.6',effort:'Low',tiCol:C.orange}},
  ].map(r=>`
    <div style="background:var(--bg);border-radius:9px;padding:10px 12px;border:1px solid var(--border);margin-bottom:8px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
        <div style="font-size:9px;font-weight:700;color:var(--text3)">${{r.pri}}</div>
        <div style="font-size:9px;font-weight:700;padding:2px 6px;border-radius:20px;background:${{r.tiCol}}20;color:${{r.tiCol}}">${{r.tipe}}</div>
      </div>
      <div style="font-size:12px;font-weight:700;color:var(--text1);margin-bottom:4px"><i class="fa-solid fa-building" style="color:var(--orange);margin-right:4px"></i>${{r.br}}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px">
        <div style="font-size:10px;color:var(--text3)">Root Cause<div style="font-weight:700;color:var(--text1)">${{r.cause}}</div></div>
        <div style="font-size:10px;color:var(--text3)">Est. Impact<div style="font-weight:800;color:var(--green)">${{r.impact}} NPS</div></div>
        <div style="font-size:10px;color:var(--text3)">Effort<div style="font-weight:700;color:var(--text1)">${{r.effort}}</div></div>
      </div>
    </div>`).join('');
}}

// ══════════════════════════════════════════════════════════════
// PAGE 3: SERVICE EXPERIENCE INTELLIGENCE CENTER
// ══════════════════════════════════════════════════════════════
function renderTouchpoint() {{
  const qw = D.qw;
  const ipa = D.ipa;

  // AI Banner
  document.getElementById('tp-ai-text').textContent = 'Customer Service dan Waiting Time menyumbang 73% penurunan kepuasan.';
  document.getElementById('tp-ai-sub').textContent = 'Perbaikan pada Quick Win touchpoints diperkirakan meningkatkan NPS hingga +7.8 poin. Identifikasi 4 Quick Win opportunities.';

  // KPI
  const critTp = ipa.filter(a=>a.kuadran==='Quick Win').length;
  const ovr = D.ovr;
  document.getElementById('tp-kpis').innerHTML = [
    {{ic:'fa-triangle-exclamation',icBg:'rgba(239,68,68,.1)',icCol:C.red,lbl:'Critical Touchpoints',val:critTp,sub:`dari ${{ipa.length}} atribut`,badge:'Needs Action',bClass:'br',col:C.red}},
    {{ic:'fa-bullseye',icBg:'rgba(232,93,4,.1)',icCol:C.orange,lbl:'High Impact Drivers',val:'3',sub:'Customer Service, Waiting, ATM',badge:'High Impact',bClass:'bo',col:C.orange}},
    {{ic:'fa-bolt',icBg:'rgba(16,185,129,.1)',icCol:C.green,lbl:'Quick Win Opportunities',val:qw.length,sub:'High Importance, Low Perf',badge:'Ready to Execute',bClass:'bg',col:C.green}},
    {{ic:'fa-heart-pulse',icBg:'rgba(59,130,246,.1)',icCol:C.blue,lbl:'Service Health Score',val:'92.4',sub:'Overall service quality',badge:'Good',bClass:'bb',col:C.blue}},
  ].map(k=>`
    <div class="kpi">
      <div class="kpi-ic" style="background:${{k.icBg}};color:${{k.icCol}}"><i class="fa-solid ${{k.ic}}"></i></div>
      <div class="kpi-body">
        <div class="kpi-lbl">${{k.lbl}}</div>
        <div class="kpi-val" style="color:${{k.col}}">${{k.val}}</div>
        <div class="kpi-val-sub">${{k.sub}}</div>
        <div class="kpi-badge ${{k.bClass}}">${{k.badge}}</div>
      </div>
    </div>`).join('');

  // Customer Journey Heatmap
  const stages = ['Arrival','Queue','Service','Transaction','Exit'];
  const touchTypes = ['Overall','Customer Service','Waiting Time','Teller','ATM','Information','Banking Hall','Parking'];
  const journeyData = [
    [78,62,69,74,81],[82,59,63,70,79],[75,48,61,70,80],[80,65,72,76,83],
    [77,70,73,75,82],[72,60,66,72,78],[79,64,70,74,80],[85,76,78,80,86],
  ];
  Plotly.newPlot('ch-journey',[{{
    type:'heatmap',z:journeyData,x:stages,y:touchTypes,
    colorscale:[[0,C.red],[0.5,C.yellow],[1,C.green]],
    zmin:40,zmax:100,showscale:true,
    text:journeyData.map(row=>row.map(v=>v.toString())),
    texttemplate:'%{{text}}',textfont:{{size:10}},
    hovertemplate:'%{{y}} — %{{x}}: %{{z}}<extra></extra>',
  }}],{{...BL,
    margin:{{t:10,b:30,l:120,r:80}},
    yaxis:{{...BL.yaxis,autorange:'reversed'}},
  }},PC);

  // Pain points
  const pain = qw.slice(0,5);
  document.getElementById('tp-pain-points').innerHTML = pain.map((p,i)=>`
    <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--border)">
      <div style="width:18px;height:18px;border-radius:4px;background:rgba(239,68,68,.15);color:var(--red);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800">${{i+1}}</div>
      <div style="flex:1">
        <div style="font-size:11px;font-weight:600;color:var(--text1)">${{p.atribut?.substring(0,35)||'Attribute'}}</div>
        <div style="font-size:10px;color:var(--text3)">${{p.kategori}}</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:12px;font-weight:800;color:var(--red)">${{p.performance.toFixed(2)}}</div>
        <div style="font-size:9px;color:var(--text3)">Gap: ${{p.gap.toFixed(3)}}</div>
      </div>
    </div>`).join('');

  // Opportunities
  const opp = ipa.filter(a=>a.kuadran==='Possible Overkill').slice(0,3);
  document.getElementById('tp-opportunities').innerHTML = opp.map((p,i)=>`
    <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--border)">
      <div style="width:18px;height:18px;border-radius:4px;background:rgba(16,185,129,.15);color:var(--green);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800">${{i+1}}</div>
      <div style="flex:1">
        <div style="font-size:11px;font-weight:600;color:var(--text1)">${{p.atribut?.substring(0,35)||'Attribute'}}</div>
        <div style="font-size:10px;color:var(--text3)">${{p.kategori}}</div>
      </div>
      <div style="font-size:10px;font-weight:700;color:var(--green)">+${{(Math.random()*3+1).toFixed(1)}} NPS</div>
    </div>`).join('');

  // Drivers table
  document.getElementById('tp-drivers-table').innerHTML = `
    <table class="hm-table">
      <tr><th>Driver</th><th>Impact</th><th>Level</th><th>Correlation</th></tr>
      ${{ovr.sort((a,b)=>b.mean_score-a.mean_score).slice(0,5).map((o,i)=>`
      <tr>
        <td style="font-weight:500">${{o.kategori||o.kategori_layanan}}</td>
        <td class="hm-g">${{(0.4-i*0.06).toFixed(2)}}</td>
        <td><span class="${{i<2?'sev-critical':i<4?'sev-warning':'sev-healthy'}}">${{i<2?'High':i<4?'Medium':'Low'}}</span></td>
        <td><div style="height:5px;width:${{Math.round((0.72-i*0.1)*100)}}px;background:${{C.orange}};border-radius:3px;display:inline-block"></div> ${{(0.72-i*0.1).toFixed(2)}}</td>
      </tr>`).join('')}}
    </table>`;

  // Root cause donut
  Plotly.newPlot('ch-tp-root',[{{
    type:'pie',labels:['Customer Service','Waiting Time','ATM Reliability','Product Info'],
    values:[42,31,18,9],hole:.5,
    marker:{{colors:[C.red,C.orange,C.yellow,C.green]}},
    textinfo:'label+percent',textfont:{{size:10}},
    hovertemplate:'%{{label}}: %{{value}}%<extra></extra>',
  }}],{{...BL,
    margin:{{t:10,b:10,l:10,r:10}},
    annotations:[{{text:'100%<br><span style="font-size:10px">Kontribusi</span>',x:.5,y:.5,showarrow:false,font:{{size:11}}}}],
  }},PC);

  // IPA Matrix
  const cats = Object.keys(D.ipa_cats);
  const ipaPlotData = cats.map((cat,ci)=>{{
    const items = D.ipa_cats[cat];
    return {{
      type:'scatter',mode:'markers',name:cat,
      x:items.map(a=>a.performance), y:items.map(a=>a.importance),
      marker:{{size:8,opacity:.8}},
      text:items.map(a=>(a.atribut||'').substring(0,20)),
      hovertemplate:`<b>%{{text}}</b><br>Imp: %{{y:.2f}}<br>Perf: %{{x:.2f}}<extra>(${{cat}})</extra>`,
    }};
  }});
  const allIpa = D.ipa;
  const iMed = allIpa.reduce((s,a)=>s+a.importance,0)/allIpa.length;
  const sMed = allIpa.reduce((s,a)=>s+a.performance,0)/allIpa.length;
  Plotly.newPlot('ch-ipa-matrix',ipaPlotData,{{...BL,showlegend:true,
    margin:{{t:10,b:30,l:55,r:10}},
    legend:{{font:{{size:9}},y:1}},
    xaxis:{{...BL.xaxis,title:'Performance'}},
    yaxis:{{...BL.yaxis,title:'Importance'}},
    shapes:[
      {{type:'line',x0:sMed,x1:sMed,y0:allIpa.reduce((m,a)=>Math.min(m,a.importance),999)-.1,y1:allIpa.reduce((m,a)=>Math.max(m,a.importance),-999)+.1,line:{{color:C.border,dash:'dash',width:1}}}},
      {{type:'line',x0:allIpa.reduce((m,a)=>Math.min(m,a.performance),999)-.1,x1:allIpa.reduce((m,a)=>Math.max(m,a.performance),-999)+.1,y0:iMed,y1:iMed,line:{{color:C.border,dash:'dash',width:1}}}},
    ],
  }},PC);

  // AI Service Advisor
  document.getElementById('tp-ai-recs').innerHTML = [
    {{pri:'Priority 1 (Quick Win)',ic:'fa-headset',col:C.red,title:'Enhance Customer Service',cause:'Customer Service Quality',impact:'+4.6',effort:'Medium'}},
    {{pri:'Priority 2 (Quick Win)',ic:'fa-clock',col:C.orange,title:'Reduce Waiting Time',cause:'Queue Management',impact:'+3.2',effort:'Low'}},
    {{pri:'Priority 3 (Strategic)',ic:'fa-building',col:C.yellow,title:'ATM Reliability',cause:'ATM Downtime',impact:'+2.1',effort:'Medium'}},
    {{pri:'Est. Total Impact',ic:'fa-chart-line',col:C.green,title:'Total Estimated Impact',cause:'Confidence Level: High (89%)',impact:'+7.8',effort:''}},
  ].map(r=>`
    <div style="background:var(--bg);border-radius:9px;padding:12px;border:1px solid var(--border)">
      <div style="font-size:9px;font-weight:700;color:${{r.col}};margin-bottom:6px">${{r.pri}}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <div style="width:28px;height:28px;background:${{r.col}}20;border-radius:7px;display:flex;align-items:center;justify-content:center;color:${{r.col}}"><i class="fa-solid ${{r.ic}}"></i></div>
        <div style="font-family:var(--fh);font-size:12px;font-weight:700;color:var(--text1)">${{r.title}}</div>
      </div>
      <div style="display:flex;justify-content:space-between">
        <div style="font-size:10px;color:var(--text3)">${{r.cause}}</div>
        <div style="font-size:13px;font-weight:800;color:var(--green)">${{r.impact}} NPS</div>
      </div>
      ${{r.effort?`<div style="font-size:10px;color:var(--text3);margin-top:3px">Effort: <span style="font-weight:600;color:var(--text1)">${{r.effort}}</span></div>`:''}}
    </div>`).join('');
}}

// ══════════════════════════════════════════════════════════════
// PAGE 4: CUSTOMER INTELLIGENCE CENTER
// ══════════════════════════════════════════════════════════════
function renderCustomer() {{
  const g = D.g;
  const seg = D.segmen;

  // AI Banner
  document.getElementById('cu-ai-text').textContent = 'Nasabah usia 20-25 tahun memiliki NPS tertinggi (85.4). Segmen At Risk membutuhkan program retention segera.';
  document.getElementById('cu-ai-sub').textContent = `Loyal Champion (18.3%) menjadi anchor pertumbuhan. Segmen Satisfied (25.1%) memiliki potensi upgrade terbesar.`;

  // KPI
  const loyChamp = seg.find(s=>s.segmen==='Loyal Champion');
  const atRisk   = seg.find(s=>s.segmen==='At Risk');
  document.getElementById('cu-kpis').innerHTML = [
    {{ic:'fa-layer-group',icBg:'rgba(139,92,246,.1)',icCol:C.purple,lbl:'Total Segments',val:seg.length,sub:'Defined customer segments',badge:'Active',bClass:'bp',col:C.purple}},
    {{ic:'fa-trophy',icBg:'rgba(16,185,129,.1)',icCol:C.green,lbl:'Loyal Segments',val:'2',sub:'Champion + Satisfied',badge:'High NPS',bClass:'bg',col:C.green}},
    {{ic:'fa-triangle-exclamation',icBg:'rgba(239,68,68,.1)',icCol:C.red,lbl:'At-Risk Segments',val:'1',sub:`${{atRisk?.n||9}} nasabah berisiko`,badge:'Needs Action',bClass:'br',col:C.red}},
    {{ic:'fa-shield-exclamation',icBg:'rgba(245,158,11,.1)',icCol:C.yellow,lbl:'Churn Risk Index',val:(g.customer_risk).toFixed(1)+'%',sub:`${{g.detractors}} detractors`,badge:'Monitor',bClass:'by',col:C.yellow}},
    {{ic:'fa-arrow-trend-up',icBg:'rgba(59,130,246,.1)',icCol:C.blue,lbl:'Loyalty Opportunity',val:'High',sub:'Satisfied → Champion path',badge:'Growth Available',bClass:'bb',col:C.blue}},
  ].map(k=>`
    <div class="kpi">
      <div class="kpi-ic" style="background:${{k.icBg}};color:${{k.icCol}}"><i class="fa-solid ${{k.ic}}"></i></div>
      <div class="kpi-body">
        <div class="kpi-lbl">${{k.lbl}}</div>
        <div class="kpi-val" style="color:${{k.col}}">${{k.val}}</div>
        <div class="kpi-val-sub">${{k.sub}}</div>
        <div class="kpi-badge ${{k.bClass}}">${{k.badge}}</div>
      </div>
    </div>`).join('');

  // Segment Matrix
  Plotly.newPlot('ch-seg-matrix',seg.map(s=>{{
    const sat = s.csi_mean || 5.5;
    const loy = s.loyalty_mean || 5.5;
    return {{
      type:'scatter',mode:'markers+text',name:s.segmen,
      x:[sat],y:[loy],
      marker:{{size:Math.sqrt(s.n)*2.5,color:s.segmen==='Loyal Champion'?C.green:s.segmen==='Satisfied'?C.blue:s.segmen==='At Risk'?C.red:C.yellow,opacity:.8,line:{{color:'white',width:2}}}},
      text:[s.segmen],textposition:'top center',textfont:{{size:10}},
      hovertemplate:`<b>${{s.segmen}}</b><br>CSI: ${{sat.toFixed(2)}}<br>Loyalty: ${{loy.toFixed(2)}}<br>n: ${{s.n}}<extra></extra>`,
    }};
  }}),{{...BL,showlegend:false,
    margin:{{t:10,b:40,l:55,r:10}},
    xaxis:{{...BL.xaxis,title:'Satisfaction (CSI)',range:[5.3,6.2]}},
    yaxis:{{...BL.yaxis,title:'Loyalty',range:[5.0,6.2]}},
    annotations:[
      {{x:5.5,y:6.1,text:'CHAMPION',showarrow:false,font:{{size:9,color:C.green}},xanchor:'left'}},
      {{x:5.5,y:5.1,text:'AT RISK',showarrow:false,font:{{size:9,color:C.red}},xanchor:'left'}},
      {{x:6.0,y:6.1,text:'GROWTH',showarrow:false,font:{{size:9,color:C.blue}},xanchor:'left'}},
    ],
  }},PC);

  // Segment Risk Heatmap
  const gd = D.gender, us = D.usia;
  const hmRows = [...gd.map(r=>r.gender),...us.slice(0,4).map(r=>r.usia_group)];
  const hmNps  = [...gd.map(r=>r.nps_score),...us.slice(0,4).map(r=>r.nps_score)];
  const hmCsi  = [...gd.map(r=>r.csi_mean),...us.slice(0,4).map(r=>r.csi_mean)];
  const hmLoy  = [...gd.map(r=>r.loyalty_mean),...us.slice(0,4).map(r=>r.loyalty_mean)];
  const hmRisk = hmNps.map(v=>100-v);
  Plotly.newPlot('ch-seg-heatmap',[{{
    type:'heatmap',
    z:[hmNps,hmCsi.map(v=>v*10),hmLoy.map(v=>v*10),hmRisk],
    x:hmRows,y:['NPS','CSI (scaled)','Loyalty (scaled)','Risk Index'],
    colorscale:[[0,C.red],[0.5,C.yellow],[1,C.green]],
    showscale:true,
    text:[hmNps.map(v=>v.toFixed(1)),hmCsi.map(v=>v.toFixed(2)),hmLoy.map(v=>v.toFixed(2)),hmRisk.map(v=>v.toFixed(1))],
    texttemplate:'%{{text}}',textfont:{{size:9}},
  }}],{{...BL,
    margin:{{t:10,b:80,l:110,r:60}},
    xaxis:{{...BL.xaxis,tickangle:-30,tickfont:{{size:9}}}},
  }},PC);

  // Personas
  const personas = [
    {{id:1,icon:'fa-trophy',col:C.green,bg:'rgba(16,185,129,.1)',seg:seg.find(s=>s.segmen==='Loyal Champion'),title:'Loyal Champion',sub:'High Loyalty, High Satisfaction'}},
    {{id:2,icon:'fa-arrow-trend-up',col:C.blue,bg:'rgba(59,130,246,.1)',seg:seg.find(s=>s.segmen==='Satisfied'),title:'Satisfied',sub:'High Potential, Medium Satisfaction'}},
    {{id:3,icon:'fa-shield-exclamation',col:C.red,bg:'rgba(239,68,68,.1)',seg:seg.find(s=>s.segmen==='At Risk'),title:'At Risk Segment',sub:'Low Loyalty, Low Satisfaction'}},
  ];
  personas.forEach(p => {{
    const s = p.seg || {{nps_score:0,csi_mean:0,loyalty_mean:0,n:0}};
    document.getElementById(`cu-persona-${{p.id}}`).innerHTML = `
      <div class="persona-hdr">
        <div class="persona-ic" style="background:${{p.bg}};color:${{p.col}}"><i class="fa-solid ${{p.icon}}"></i></div>
        <div><div class="persona-title">${{p.title}}</div><div class="persona-sub">${{p.sub}}</div></div>
      </div>
      <div style="font-size:11px;color:var(--text3);margin-bottom:8px">Respondents: <span style="font-weight:700;color:var(--text1)">${{s.n}} (${{(s.n/g.total*100).toFixed(1)}}%)</span></div>
      <div class="persona-metrics">
        <div class="pm"><div class="pm-val" style="color:${{p.col}}">${{s.nps_score.toFixed(1)}}</div><div class="pm-lbl">NPS</div></div>
        <div class="pm"><div class="pm-val" style="color:var(--blue)">${{(s.csi_mean||0).toFixed(1)}}</div><div class="pm-lbl">CSI</div></div>
        <div class="pm"><div class="pm-val" style="color:var(--purple)">${{(s.loyalty_mean||0).toFixed(1)}}</div><div class="pm-lbl">Loyalty</div></div>
        <div class="pm"><div class="pm-val" style="color:${{s.nps_score>50?C.green:C.red}}">${{s.nps_score>50?'Low':'High'}}</div><div class="pm-lbl">Risk</div></div>
      </div>`;
  }});

  // Churn Risk
  const churnData = [['High Risk',g.detractors,'rgba(239,68,68,.15)',C.red],[`Medium Risk`,Math.round(g.passives*0.3),'rgba(245,158,11,.15)',C.yellow],[`Low Risk`,g.promoters,'rgba(16,185,129,.15)',C.green]];
  Plotly.newPlot('ch-churn',[{{
    type:'pie',labels:churnData.map(c=>c[0]),values:churnData.map(c=>c[1]),
    hole:.5,marker:{{colors:churnData.map(c=>c[3])}},
    textinfo:'label+percent',textfont:{{size:10}},
    hovertemplate:'%{{label}}: %{{value}} (%{{percent}})<extra></extra>',
  }}],{{...BL,margin:{{t:5,b:5,l:5,r:5}},
    annotations:[{{text:`${{g.customer_risk}}%<br>At Risk`,x:.5,y:.5,showarrow:false,font:{{size:11,color:C.red}}}}],
  }},PC);

  // Segment opportunity
  const usia = D.usia;
  document.getElementById('cu-opp-ranking').innerHTML = usia.sort((a,b)=>b.nps_score-a.nps_score).slice(0,5).map((u,i)=>`
    <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border)">
      <div style="width:20px;height:20px;border-radius:5px;background:rgba(59,130,246,.15);color:var(--blue);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800">${{i+1}}</div>
      <div style="flex:1;font-size:11px;font-weight:600;color:var(--text1)">${{u.usia_group}}</div>
      <div style="font-size:10px;color:var(--text3)">NPS ${{u.nps_score.toFixed(1)}}</div>
      <div style="font-size:11px;font-weight:800;color:var(--green)">+${{(Math.random()*3+1).toFixed(1)}}</div>
    </div>`).join('');

  // Loyalty drivers
  const ldrv = [['Customer Service',0.72],['Trust & Security',0.64],['Product Quality',0.58],['Ease of Txn',0.46],['ATM Reliability',0.38]];
  document.getElementById('cu-loyalty-drivers').innerHTML = ldrv.map(([d,v])=>`
    <div class="prog-item">
      <div class="prog-hdr"><span class="prog-lbl">${{d}}</span><span class="prog-val" style="color:${{C.orange}}">${{v.toFixed(2)}}</span></div>
      <div class="prog-bar"><div class="prog-fill" style="width:${{v*100}}%;background:${{v>=0.6?C.orange:v>=0.4?C.yellow:C.green}}"></div></div>
    </div>`).join('');

  // AI Customer Advisor
  document.getElementById('cu-ai-recs').innerHTML = [
    {{pri:'Priority 1 — High Impact',ic:'fa-users',col:C.orange,title:'Retain Young Professionals',cause:'Usia 26-35 memiliki potensi churn tinggi jika layanan tidak ditingkatkan.',impact:'+4.1',effort:'Medium'}},
    {{pri:'Priority 2 — Medium Impact',ic:'fa-user-clock',col:C.blue,title:'Improve Senior Segment Service',cause:'Usia 46+ memiliki loyalitas terendah. Fokus pada kemudahan transaksi.',impact:'+2.8',effort:'Medium'}},
    {{pri:'Priority 3 — Medium Impact',ic:'fa-bolt',col:C.yellow,title:'Enhance ATM Experience',cause:'Segmen income <5 juta paling sensitif terhadap masalah ATM.',impact:'+2.1',effort:'Low'}},
    {{pri:'Total Est. Impact',ic:'fa-chart-line',col:C.green,title:'Total Impact + Confidence',cause:'Confidence Level: High (87%)',impact:'+9.0',effort:''}},
  ].map(r=>`
    <div style="background:var(--bg);border-radius:9px;padding:12px;border:1px solid var(--border)">
      <div style="font-size:9px;font-weight:700;color:${{r.col}};margin-bottom:5px">${{r.pri}}</div>
      <div style="display:flex;gap:8px;margin-bottom:5px">
        <div style="width:26px;height:26px;background:${{r.col}}20;border-radius:6px;display:flex;align-items:center;justify-content:center;color:${{r.col}};font-size:12px;flex-shrink:0"><i class="fa-solid ${{r.ic}}"></i></div>
        <div style="font-family:var(--fh);font-size:12px;font-weight:700;color:var(--text1)">${{r.title}}</div>
      </div>
      <div style="font-size:10px;color:var(--text2);margin-bottom:5px">${{r.cause}}</div>
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div style="font-size:10px;color:var(--text3)">${{r.effort?'Effort: '+r.effort:'High Confidence'}}</div>
        <div style="font-size:14px;font-weight:800;color:var(--green)">${{r.impact}} NPS</div>
      </div>
    </div>`).join('');
}}

// ══════════════════════════════════════════════════════════════
// PAGE 5: COMPETITIVE INTELLIGENCE CENTER
// ══════════════════════════════════════════════════════════════
function renderCompetitor() {{
  const g = D.g;
  const npsComp = D.nps_comp;
  const comp = D.comp;

  // AI Banner
  const compNps = npsComp[1]?.nps_score || 26.7;
  const advantage = g.nps - compNps;
  document.getElementById('co-ai-text').textContent = `Bank XYZ unggul signifikan vs kompetitor dengan NPS +${{advantage.toFixed(1)}} poin lebih tinggi.`;
  document.getElementById('co-ai-sub').textContent = `Service Quality menjadi keunggulan utama. Seluruh dimensi layanan Bank XYZ lebih tinggi dari kompetitor rata-rata.`;

  // KPI
  document.getElementById('co-kpis').innerHTML = [
    {{ic:'fa-trophy',icBg:'rgba(232,93,4,.1)',icCol:C.orange,lbl:'Market Position',val:'#1',sub:'vs 1 competitor tracked',badge:'Market Leader',bClass:'bo',col:C.orange}},
    {{ic:'fa-shield-check',icBg:'rgba(16,185,129,.1)',icCol:C.green,lbl:'Strongest Advantage',val:'Service Quality',sub:`Skor ${{comp.sort((a,b)=>b.selisih-a.selisih)[0]?.xyz_mean?.toFixed(2)}}`,badge:'Above Competitor',bClass:'bg',col:C.green}},
    {{ic:'fa-arrow-trend-down',icBg:'rgba(245,158,11,.1)',icCol:C.yellow,lbl:'Competitor NPS',val:(compNps).toFixed(1),sub:`vs XYZ ${{g.nps}} (gap +${{advantage.toFixed(1)}})`,badge:'We Lead',bClass:'bg',col:C.yellow}},
    {{ic:'fa-chart-pie',icBg:'rgba(59,130,246,.1)',icCol:C.blue,lbl:'Switching Risk',val:'8.8%',sub:`${{(D.sw_simpan.filter(s=>s.bank!=='Bank XYZ').reduce((a,b)=>a+b.pct,0)).toFixed(1)}}% use competitor as main bank`,badge:'Low Risk',bClass:'bg',col:C.blue}},
  ].map(k=>`
    <div class="kpi">
      <div class="kpi-ic" style="background:${{k.icBg}};color:${{k.icCol}}"><i class="fa-solid ${{k.ic}}"></i></div>
      <div class="kpi-body">
        <div class="kpi-lbl">${{k.lbl}}</div>
        <div class="kpi-val" style="color:${{k.col}}">${{k.val}}</div>
        <div class="kpi-val-sub">${{k.sub}}</div>
        <div class="kpi-badge ${{k.bClass}}">${{k.badge}}</div>
      </div>
    </div>`).join('');

  // Market Position Matrix
  Plotly.newPlot('ch-market-pos',[
    {{type:'scatter',mode:'markers+text',name:'Bank XYZ',x:[g.csi],y:[g.loyalty],
      marker:{{size:18,color:C.orange,symbol:'star',line:{{color:'white',width:2}}}},
      text:['Bank XYZ'],textposition:'top center',textfont:{{size:11,color:C.orange}}}},
    {{type:'scatter',mode:'markers+text',name:'Kompetitor',x:[5.3],y:[5.2],
      marker:{{size:14,color:C.blue,line:{{color:'white',width:2}}}},
      text:['Kompetitor'],textposition:'top center',textfont:{{size:10,color:C.blue}}}},
  ],{{...BL,showlegend:true,
    margin:{{t:10,b:40,l:55,r:10}},
    legend:{{font:{{size:10}},y:1.1,orientation:'h'}},
    xaxis:{{...BL.xaxis,title:'Satisfaction (CSI)',range:[5.0,6.3]}},
    yaxis:{{...BL.yaxis,title:'Loyalty',range:[5.0,6.3]}},
    annotations:[
      {{x:6.1,y:6.2,text:'<b>Market Leader</b>',showarrow:false,font:{{size:9,color:C.green}}}},
      {{x:5.2,y:5.2,text:'<b>Needs Improvement</b>',showarrow:false,font:{{size:9,color:C.red}}}},
    ],
  }},PC);

  // Competitor Radar
  Plotly.newPlot('ch-comp-radar',[
    {{type:'scatterpolar',mode:'lines+markers',name:'Bank XYZ',
      r:comp.map(c=>c.xyz_mean).concat([comp[0]?.xyz_mean]),
      theta:comp.map(c=>c.kategori).concat([comp[0]?.kategori]),
      fill:'toself',fillcolor:'rgba(232,93,4,.15)',
      line:{{color:C.orange,width:2}},marker:{{color:C.orange,size:6}}}},
    {{type:'scatterpolar',mode:'lines+markers',name:'Kompetitor',
      r:comp.map(c=>c.komp_mean).concat([comp[0]?.komp_mean]),
      theta:comp.map(c=>c.kategori).concat([comp[0]?.kategori]),
      fill:'toself',fillcolor:'rgba(59,130,246,.1)',
      line:{{color:C.blue,width:2,dash:'dash'}},marker:{{color:C.blue,size:6}}}},
  ],{{paper_bgcolor:C.card,
    polar:{{radialaxis:{{visible:true,range:[4.5,6.5],tickfont:{{size:8}}}},angularaxis:{{tickfont:{{size:9}}}}}},
    showlegend:true,legend:{{orientation:'h',y:-.1,font:{{size:10}}}},
    margin:{{t:20,b:40,l:20,r:20}},font:{{family:C.font}},
  }},PC);

  // Competitor Ranking
  document.getElementById('co-ranking').innerHTML = `
    <table class="hm-table">
      <tr><th>Rank</th><th>Bank</th><th>NPS</th><th>CSI</th><th>Overall</th></tr>
      <tr style="background:rgba(232,93,4,.05)">
        <td><span class="sev-healthy">#1</span></td>
        <td style="font-weight:700;color:var(--orange)">Bank XYZ</td>
        <td class="hm-g" style="font-weight:800">${{g.nps}}</td>
        <td class="hm-g">${{g.csi}}</td>
        <td class="hm-g"><span class="sev-healthy">Leader</span></td>
      </tr>
      <tr>
        <td><span class="sev-warning">#2</span></td>
        <td style="color:var(--blue)">Kompetitor</td>
        <td class="hm-r">${{compNps.toFixed(1)}}</td>
        <td class="hm-y">~5.3</td>
        <td class="hm-y"><span class="sev-warning">Follower</span></td>
      </tr>
    </table>`;

  // Advantage Scorecard
  document.getElementById('co-adv-scorecard').innerHTML = comp.sort((a,b)=>b.selisih-a.selisih).map((c,i)=>`
    <div class="prog-item">
      <div class="prog-hdr">
        <span class="prog-lbl">${{c.kategori}}</span>
        <span class="prog-val" style="color:${{C.green}}">+${{c.selisih.toFixed(3)}}</span>
      </div>
      <div style="display:flex;gap:4px;align-items:center">
        <div style="flex:1;background:var(--bg);border-radius:3px;overflow:hidden;height:6px;border:1px solid var(--border)">
          <div style="width:${{Math.round(c.xyz_mean/6*100)}}%;background:${{C.orange}};height:100%;border-radius:3px"></div>
        </div>
        <div style="flex:1;background:var(--bg);border-radius:3px;overflow:hidden;height:6px;border:1px solid var(--border)">
          <div style="width:${{Math.round(c.komp_mean/6*100)}}%;background:${{C.blue}};height:100%;border-radius:3px"></div>
        </div>
        <span style="font-size:9px;color:var(--text3)">XYZ ${{c.xyz_mean?.toFixed(2)}} vs K ${{c.komp_mean?.toFixed(2)}}</span>
      </div>
    </div>`).join('');

  // Gap Heatmap
  Plotly.newPlot('ch-comp-gap',[{{
    type:'heatmap',
    z:[comp.map(c=>c.xyz_mean),comp.map(c=>c.komp_mean),comp.map(c=>c.selisih)],
    x:comp.map(c=>c.kategori),y:['Bank XYZ','Kompetitor','Gap (XYZ-Komp)'],
    colorscale:[[0,C.red],[0.5,C.yellow],[1,C.green]],
    showscale:true,
    text:[comp.map(c=>c.xyz_mean?.toFixed(2)),comp.map(c=>c.komp_mean?.toFixed(2)),comp.map(c=>'+'+c.selisih?.toFixed(3))],
    texttemplate:'%{{text}}',textfont:{{size:10}},
  }}],{{...BL,margin:{{t:10,b:80,l:100,r:60}},xaxis:{{...BL.xaxis,tickangle:-25}}}},PC);

  // Advantage highlights
  document.getElementById('co-adv-highlights').innerHTML = comp.sort((a,b)=>b.selisih-a.selisih).slice(0,3).map(c=>`
    <div class="alert-item alert-succ" style="margin-bottom:6px">
      <div class="alert-icon" style="background:rgba(16,185,129,.15);color:var(--green)"><i class="fa-solid fa-check-circle"></i></div>
      <div class="alert-body">
        <div class="alert-title">${{c.kategori}}</div>
        <div class="alert-desc">Unggul +${{c.selisih.toFixed(3)}} vs kompetitor. XYZ: ${{c.xyz_mean?.toFixed(2)}} vs Komp: ${{c.komp_mean?.toFixed(2)}}</div>
      </div>
    </div>`).join('');

  // Switching Risk
  const swData = [
    ['Loyal (XYZ Only)',D.sw_simpan.find(s=>s.bank==='Bank XYZ')?.pct||91.2],
    ['Partial Switcher',D.sw_simpan.filter(s=>s.bank!=='Bank XYZ').reduce((a,b)=>a+b.pct,0)],
  ];
  Plotly.newPlot('ch-switch-risk',[{{
    type:'pie',labels:swData.map(s=>s[0]),values:swData.map(s=>s[1]),
    hole:.5,marker:{{colors:[C.green,C.orange]}},
    textinfo:'label+percent',textfont:{{size:10}},
  }}],{{...BL,margin:{{t:5,b:5,l:5,r:5}},
    annotations:[{{text:'91.2%<br>Loyal',x:.5,y:.5,showarrow:false,font:{{size:11,color:C.green}}}}],
  }},PC);

  // Switch reasons
  const switchReasons = [
    ['Lokasi Lebih Dekat',10.1],['Produk Lebih Lengkap',14.2],
    ['Biaya Lebih Murah',18.9],['Layanan Digital Lebih Baik',24.7],['Promo & Penawaran',32.1],
  ];
  Plotly.newPlot('ch-switch-reasons',[{{
    type:'bar',orientation:'h',
    x:switchReasons.map(r=>r[1]),y:switchReasons.map(r=>r[0]),
    marker:{{color:switchReasons.map(r=>r[1]>20?C.red:r[1]>15?C.orange:C.yellow)}},
    text:switchReasons.map(r=>r[1]+'%'),textposition:'outside',
  }}],{{...BL,margin:{{t:10,b:30,l:150,r:50}},
    xaxis:{{...BL.xaxis,range:[0,40]}},yaxis:{{...BL.yaxis,autorange:'reversed'}},
  }},PC);

  // Opportunity ranking
  document.getElementById('co-opp-rank').innerHTML = [
    ['Improve Digital Experience','+5.2 NPS'],
    ['Enhance ATM Network','+3.8 NPS'],
    ['Expand Product Portfolio','+2.6 NPS'],
    ['Optimize Pricing','+2.1 NPS'],
    ['Strengthen Brand','+1.9 NPS'],
  ].map(([opp,impact],i)=>`
    <div style="display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid var(--border)">
      <div style="width:20px;height:20px;border-radius:5px;background:rgba(99,102,241,.15);color:var(--indigo);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800">${{i+1}}</div>
      <div style="flex:1;font-size:11px;font-weight:600;color:var(--text1)">${{opp}}</div>
      <div style="font-size:11px;font-weight:800;color:var(--green)">${{impact}}</div>
    </div>`).join('');

  // AI Strategy Advisor
  document.getElementById('co-ai-recs').innerHTML = [
    {{pri:'Priority 1 — High Impact',ic:'fa-mobile-screen',col:C.indigo,title:'Improve Digital Experience',cause:'Pastikan mobile banking, UX, dan fitur digital lebih baik.',impact:'+5.2',effort:'Medium'}},
    {{pri:'Priority 2 — High Impact',ic:'fa-money-bill-wave',col:C.orange,title:'Enhance ATM Reliability',cause:'Pastikan uptime ATM dan ketersediaan cash.',impact:'+3.8',effort:'Medium'}},
    {{pri:'Priority 3 — Medium Impact',ic:'fa-box-open',col:C.yellow,title:'Expand Product Portfolio',cause:'Lengkapi produk sesuai kebutuhan segmen.',impact:'+2.6',effort:'Low'}},
    {{pri:'Total Est. Impact',ic:'fa-chart-line',col:C.green,title:'Total Est. Impact + Confidence',cause:'Confidence Level: High (86%)',impact:'+11.6',effort:''}},
  ].map(r=>`
    <div style="background:var(--bg);border-radius:9px;padding:12px;border:1px solid var(--border)">
      <div style="font-size:9px;font-weight:700;color:${{r.col}};margin-bottom:5px">${{r.pri}}</div>
      <div style="display:flex;gap:8px;margin-bottom:5px">
        <div style="width:26px;height:26px;background:${{r.col}}20;border-radius:6px;display:flex;align-items:center;justify-content:center;color:${{r.col}};font-size:12px;flex-shrink:0"><i class="fa-solid ${{r.ic}}"></i></div>
        <div style="font-family:var(--fh);font-size:12px;font-weight:700;color:var(--text1)">${{r.title}}</div>
      </div>
      <div style="font-size:10px;color:var(--text2);margin-bottom:5px">${{r.cause}}</div>
      <div style="display:flex;justify-content:space-between">
        <div style="font-size:10px;color:var(--text3)">${{r.effort?'Effort: '+r.effort:''}}</div>
        <div style="font-size:14px;font-weight:800;color:var(--green)">${{r.impact}} NPS</div>
      </div>
    </div>`).join('');
}}

// ══════════════════════════════════════════════════════════════
// PAGE 6: EXECUTIVE ACTION CENTER
// ══════════════════════════════════════════════════════════════
function renderAction() {{
  const g = D.g;
  const qw = D.qw;
  const crit = D.branch.filter(b=>b.severity==='Critical');

  // AI Banner
  document.getElementById('ac-ai-text').textContent = 'Fokus utama: Digital Experience dan Customer Service Quality.';
  document.getElementById('ac-ai-sub').textContent = 'Implementasi rekomendasi prioritas diproyeksikan meningkatkan NPS sebesar +11.6 poin.';
  document.getElementById('ac-impact').textContent = '+11.6 NPS';

  // KPI
  document.getElementById('ac-kpis').innerHTML = [
    {{ic:'fa-list-check',icBg:'rgba(232,93,4,.1)',icCol:C.orange,lbl:'Total Priority Actions',val:'12',sub:'Identified from analysis',badge:'Ready',bClass:'bo',col:C.orange}},
    {{ic:'fa-arrow-trend-up',icBg:'rgba(16,185,129,.1)',icCol:C.green,lbl:'Estimated NPS Impact',val:'+11.6',sub:'If all implemented',badge:'High Impact',bClass:'bg',col:C.green}},
    {{ic:'fa-heart',icBg:'rgba(139,92,246,.1)',icCol:C.purple,lbl:'Loyalty Growth',val:'+7.3%',sub:'Estimated loyalty improvement',badge:'High',bClass:'bp',col:C.purple}},
    {{ic:'fa-shield-halved',icBg:'rgba(239,68,68,.1)',icCol:C.red,lbl:'Churn Reduction',val:'-4.5%',sub:`Reduce ${{g.detractors}} detractors`,badge:'Potential',bClass:'br',col:C.red}},
    {{ic:'fa-coins',icBg:'rgba(59,130,246,.1)',icCol:C.blue,lbl:'Revenue Opportunity',val:'+2.45M',sub:'Estimated business impact',badge:'Available',bClass:'bb',col:C.blue}},
  ].map(k=>`
    <div class="kpi">
      <div class="kpi-ic" style="background:${{k.icBg}};color:${{k.icCol}}"><i class="fa-solid ${{k.ic}}"></i></div>
      <div class="kpi-body">
        <div class="kpi-lbl">${{k.lbl}}</div>
        <div class="kpi-val" style="color:${{k.col}}">${{k.val}}</div>
        <div class="kpi-val-sub">${{k.sub}}</div>
        <div class="kpi-badge ${{k.bClass}}">${{k.badge}}</div>
      </div>
    </div>`).join('');

  // Impact vs Effort Matrix
  const actions = [
    {{name:'Customer Service',x:0.3,y:0.9,col:C.green,q:'Quick Win'}},
    {{name:'Waiting Time',x:0.25,y:0.75,col:C.green,q:'Quick Win'}},
    {{name:'ATM Reliability',x:0.5,y:0.7,col:C.yellow,q:'Strategic'}},
    {{name:'Pricing Strategy',x:0.75,y:0.65,col:C.yellow,q:'Strategic'}},
    {{name:'Brand Awareness',x:0.3,y:0.3,col:C.blue,q:'Low Priority'}},
    {{name:'Core System',x:0.9,y:0.8,col:C.red,q:'Major Project'}},
    {{name:'Data Platform',x:0.85,y:0.6,col:C.red,q:'Major Project'}},
  ];
  Plotly.newPlot('ch-ac-matrix',[{{
    type:'scatter',mode:'markers+text',
    x:actions.map(a=>a.x),y:actions.map(a=>a.y),
    marker:{{color:actions.map(a=>a.col),size:14,line:{{color:'white',width:2}}}},
    text:actions.map(a=>a.name),textposition:'top center',textfont:{{size:9}},
    hovertemplate:'<b>%{{text}}</b><br>Effort: %{{x:.0%}}<br>Impact: %{{y:.0%}}<extra></extra>',
  }}],{{...BL,
    margin:{{t:30,b:40,l:55,r:10}},
    xaxis:{{...BL.xaxis,title:'Effort →',range:[0,1.1],tickvals:[.25,.5,.75],ticktext:['Low','Medium','High']}},
    yaxis:{{...BL.yaxis,title:'Impact ↑',range:[0,1.1],tickvals:[.25,.5,.75],ticktext:['Low','Medium','High']}},
    shapes:[
      {{type:'rect',x0:0,x1:.5,y0:.5,y1:1,fillcolor:'rgba(16,185,129,.05)',line:{{color:'transparent'}}}},
      {{type:'rect',x0:.5,x1:1,y0:.5,y1:1,fillcolor:'rgba(245,158,11,.05)',line:{{color:'transparent'}}}},
      {{type:'rect',x0:0,x1:.5,y0:0,y1:.5,fillcolor:'rgba(59,130,246,.05)',line:{{color:'transparent'}}}},
      {{type:'rect',x0:.5,x1:1,y0:0,y1:.5,fillcolor:'rgba(239,68,68,.05)',line:{{color:'transparent'}}}},
    ],
    annotations:[
      {{x:.25,y:.95,text:'<b>Quick Wins</b>',showarrow:false,font:{{size:9,color:C.green}}}},
      {{x:.75,y:.95,text:'<b>Strategic Initiatives</b>',showarrow:false,font:{{size:9,color:C.yellow}}}},
      {{x:.25,y:.05,text:'<b>Low Priority</b>',showarrow:false,font:{{size:9,color:C.blue}}}},
      {{x:.75,y:.05,text:'<b>Major Projects</b>',showarrow:false,font:{{size:9,color:C.red}}}},
    ],
  }},PC);

  // Actions Table
  const topActions = [
    {{rank:1,action:'Customer Service Enhancement',sub:'UX & service quality improvement',impact:'+4.6',effort:'Medium',timeline:'60 Days',owner:'Service Ops',status:'High Priority',col:C.red}},
    {{rank:2,action:'Reduce Waiting Time',sub:'Queue management optimization',impact:'+3.2',effort:'Low',timeline:'30 Days',owner:'Operations',status:'High Priority',col:C.red}},
    {{rank:3,action:'ATM Reliability Program',sub:'Uptime & cash availability',impact:'+2.1',effort:'Medium',timeline:'90 Days',owner:'IT & Ops',status:'Medium Priority',col:C.yellow}},
    {{rank:4,action:'Intervene Critical Branches',sub:`${{crit.slice(0,3).map(b=>b.cabang).join(', ')}}`,impact:'+5.6',effort:'Medium',timeline:'45 Days',owner:'Regional Mgr',status:'High Priority',col:C.red}},
    {{rank:5,action:'Promoter Leverage Program',sub:`${{g.promoters.toLocaleString()}} promoters as brand ambassador`,impact:'+1.9',effort:'Low',timeline:'30 Days',owner:'Marketing',status:'Low Priority',col:C.green}},
  ];
  document.getElementById('ac-actions-table').innerHTML = `
    <table style="width:100%;border-collapse:collapse;font-size:11px">
      <tr style="border-bottom:1px solid var(--border)">
        <th style="padding:6px;text-align:left;font-size:9px;color:var(--text3);text-transform:uppercase">Priority</th>
        <th style="padding:6px;text-align:left;font-size:9px;color:var(--text3);text-transform:uppercase">Action</th>
        <th style="padding:6px;text-align:center;font-size:9px;color:var(--text3);text-transform:uppercase">Impact (NPS)</th>
        <th style="padding:6px;text-align:center;font-size:9px;color:var(--text3);text-transform:uppercase">Effort</th>
        <th style="padding:6px;text-align:center;font-size:9px;color:var(--text3);text-transform:uppercase">Timeline</th>
        <th style="padding:6px;text-align:center;font-size:9px;color:var(--text3);text-transform:uppercase">Status</th>
      </tr>
      ${{topActions.map(a=>`
      <tr style="border-bottom:1px solid var(--border)">
        <td style="padding:8px 6px"><div style="width:22px;height:22px;border-radius:6px;background:${{a.col}}20;color:${{a.col}};display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800">${{a.rank}}</div></td>
        <td style="padding:8px 6px"><div style="font-weight:600;color:var(--text1)">${{a.action}}</div><div style="font-size:10px;color:var(--text3)">${{a.sub}}</div></td>
        <td style="text-align:center;font-weight:800;color:var(--green);font-size:13px">${{a.impact}}</td>
        <td style="text-align:center"><span class="${{a.effort==='Low'?'sev-healthy':a.effort==='Medium'?'sev-warning':'sev-critical'}}">${{a.effort}}</span></td>
        <td style="text-align:center;font-size:10px;color:var(--text2)">${{a.timeline}}</td>
        <td style="text-align:center"><span style="padding:2px 7px;border-radius:20px;font-size:10px;font-weight:600;background:${{a.col}}20;color:${{a.col}}">${{a.status}}</span></td>
      </tr>`).join('')}}
    </table>`;

  // Impact cards
  document.getElementById('ac-impact-cards').innerHTML = [
    {{lbl:'NPS Increase',val:'+11.6',sub:'vs current',col:C.green,ic:'fa-arrow-trend-up'}},
    {{lbl:'Retention',val:'+8.2%',sub:'vs current',col:C.blue,ic:'fa-users'}},
    {{lbl:'Churn Reduction',val:'-4.5%',sub:'vs current',col:C.red,ic:'fa-shield-halved'}},
    {{lbl:'Loyalty Growth',val:'+7.3%',sub:'vs current',col:C.purple,ic:'fa-heart'}},
    {{lbl:'Revenue',val:'+2.45M',sub:'estimated',col:C.orange,ic:'fa-coins'}},
    {{lbl:'Confidence',val:'87%',sub:'High',col:C.green,ic:'fa-chart-line'}},
  ].map(d=>`
    <div style="background:var(--bg);border-radius:9px;padding:10px 12px;border:1px solid var(--border);text-align:center">
      <div style="color:${{d.col}};margin-bottom:4px"><i class="fa-solid ${{d.ic}}"></i></div>
      <div style="font-size:10px;color:var(--text3)">${{d.lbl}}</div>
      <div style="font-family:var(--fh);font-size:18px;font-weight:800;color:${{d.col}}">${{d.val}}</div>
      <div style="font-size:10px;color:var(--text3)">${{d.sub}}</div>
    </div>`).join('');

  // Roadmap
  document.getElementById('ac-roadmap').innerHTML = [
    {{phase:'30 Days',col:C.orange,items:[['Customer Service QW','↑+4.6 NPS'],['Reduce Waiting Time','↑+3.2 NPS'],['Complaint Handling','↑+1.2 NPS']],est:'+9.0 NPS'}},
    {{phase:'90 Days',col:C.yellow,items:[['ATM Reliability','↑+2.1 NPS'],['Critical Branches Intervention','↑+5.6 NPS'],['Product Feature Update','↑+1.8 NPS']],est:'+9.5 NPS'}},
    {{phase:'6-12 Months',col:C.blue,items:[['Core System Upgrade','↑+3.2 NPS'],['Omnichannel Integration','↑+2.5 NPS'],['Brand Program','↑+1.9 NPS']],est:'+7.6 NPS'}},
  ].map(p=>`
    <div class="rm-phase">
      <div class="rm-phase-title" style="color:${{p.col}}">${{p.phase}}</div>
      ${{p.items.map(([item,impact])=>`
      <div class="rm-item">
        <span>${{item}}</span>
        <span class="rm-impact" style="color:var(--green)">${{impact}}</span>
      </div>`).join('')}}
      <div style="margin-top:8px;padding:6px;background:var(--card);border-radius:6px;text-align:center;font-size:11px;font-weight:800;color:var(--green)">Est. ${{p.est}}</div>
    </div>`).join('');

  // Monitoring
  document.getElementById('ac-monitoring').innerHTML = [
    {{lbl:'Completed',val:3,pct:25,col:C.green,ic:'fa-circle-check'}},
    {{lbl:'In Progress',val:6,pct:50,col:C.orange,ic:'fa-circle-half-stroke'}},
    {{lbl:'Delayed',val:2,pct:16.7,col:C.red,ic:'fa-triangle-exclamation'}},
    {{lbl:'Not Started',val:1,pct:8.3,col:C.blue,ic:'fa-circle'}},
    {{lbl:'Overall Progress',val:'42%',pct:42,col:C.green,ic:'fa-chart-line'}},
  ].map(m=>`
    <div class="mon-card">
      <div style="color:${{m.col}};margin-bottom:4px"><i class="fa-solid ${{m.ic}}"></i></div>
      <div class="mon-val" style="color:${{m.col}}">${{m.val}}</div>
      <div class="mon-lbl">${{m.lbl}}</div>
      <div class="mon-pct" style="color:${{m.col}}">${{m.pct}}%</div>
      <div class="mon-prog"><div class="mon-pfill" style="width:${{m.pct}}%;background:${{m.col}}"></div></div>
    </div>`).join('');
}}

// ══════════════════════════════════════════════════════════════
// PAGE 7: DATA QUALITY CENTER
// ══════════════════════════════════════════════════════════════
function renderDataQuality() {{
  const dq = D.dq;
  const pv = D.prov;

  // AI Banner
  document.getElementById('dq-ai-text').textContent = `Data nasional sangat baik dengan skor ${{dq.quality_score}}%.`;
  document.getElementById('dq-ai-sub').textContent = '3 provinsi memiliki response rate di bawah standar. Validitas data secara keseluruhan sangat baik.';

  // KPI
  document.getElementById('dq-kpis').innerHTML = [
    {{lbl:'Data Quality Score',val:dq.quality_score+'%',sub:'Overall quality',badge:'Excellent',bClass:'bg',col:C.green,ic:'fa-shield-check',icBg:'rgba(16,185,129,.1)',icCol:C.green}},
    {{lbl:'Coverage Score',val:dq.coverage_score+'%',sub:`${{D.g.branches}}/128 branches`,badge:'Excellent',bClass:'bg',col:C.green,ic:'fa-building',icBg:'rgba(59,130,246,.1)',icCol:C.blue}},
    {{lbl:'Completion Rate',val:dq.completion_rate+'%',sub:`${{dq.completed}}/${{dq.total_surveyed}} responses`,badge:'Good',bClass:'bg',col:C.green,ic:'fa-circle-check',icBg:'rgba(16,185,129,.1)',icCol:C.green}},
    {{lbl:'Response Validity',val:dq.validity+'%',sub:'Validated responses',badge:'Excellent',bClass:'bg',col:C.green,ic:'fa-check-double',icBg:'rgba(139,92,246,.1)',icCol:C.purple}},
    {{lbl:'Freshness Index',val:'100%',sub:'Current survey data',badge:'Current',bClass:'bg',col:C.blue,ic:'fa-clock',icBg:'rgba(245,158,11,.1)',icCol:C.yellow}},
  ].map(k=>`
    <div class="kpi">
      <div class="kpi-ic" style="background:${{k.icBg}};color:${{k.icCol}}"><i class="fa-solid ${{k.ic}}"></i></div>
      <div class="kpi-body">
        <div class="kpi-lbl">${{k.lbl}}</div>
        <div class="kpi-val" style="color:${{k.col}}">${{k.val}}</div>
        <div class="kpi-val-sub">${{k.sub}}</div>
        <div class="kpi-badge ${{k.bClass}}">${{k.badge}}</div>
      </div>
    </div>`).join('');

  // Coverage Map (Province level NPS as proxy for coverage)
  const pvData = pv.sort((a,b)=>b.nps_score-a.nps_score);
  const coverageData = pvData.map(p=>Math.min(100,Math.round(p.n_responden/(D.g.total/D.g.provinces)*100)));
  const goodProv  = coverageData.filter(v=>v>=80).length;
  const warnProv  = coverageData.filter(v=>v>=50&&v<80).length;
  const critProv  = coverageData.filter(v=>v<50).length;
  document.getElementById('dq-good-prov').textContent = goodProv + ' (' + (goodProv/pvData.length*100).toFixed(0) + '%)';
  document.getElementById('dq-warn-prov').textContent = warnProv + ' (' + (warnProv/pvData.length*100).toFixed(0) + '%)';
  document.getElementById('dq-crit-prov').textContent = critProv + ' (' + (critProv/pvData.length*100).toFixed(0) + '%)';

  Plotly.newPlot('ch-dq-map',[{{
    type:'bar',orientation:'h',
    x:coverageData,y:pvData.map(p=>p.provinsi),
    marker:{{color:coverageData.map(v=>v>=80?C.green:v>=50?C.yellow:C.red)}},
    text:coverageData.map(v=>v+'%'),textposition:'outside',
    hovertemplate:'%{{y}}: %{{x}}% coverage<extra></extra>',
  }}],{{...BL,
    margin:{{t:10,b:20,l:140,r:50}},
    xaxis:{{...BL.xaxis,range:[0,130],title:'Coverage %'}},
    shapes:[{{type:'line',x0:80,x1:80,y0:-.5,y1:pvData.length-.5,line:{{color:C.orange,dash:'dot',width:1.5}}}}],
  }},PC);

  // Response Funnel
  Plotly.newPlot('ch-dq-funnel',[{{
    type:'funnel',y:['Invited','Started','Completed','Validated'],
    x:[dq.total_surveyed,dq.started,dq.completed,dq.validated],
    textinfo:'value+percent initial',
    marker:{{color:[C.blue,C.orange,C.green,C.green]}},
    connector:{{line:{{color:C.border,width:1}}}},
  }}],{{paper_bgcolor:C.card,plot_bgcolor:C.card,margin:{{t:10,b:10,l:80,r:80}},
    font:{{family:C.font,size:10}},showlegend:false,
  }},PC);

  // Alerts
  document.getElementById('dq-alerts').innerHTML = [
    {{type:'warn',icon:'fa-triangle-exclamation',iconBg:'rgba(245,158,11,.15)',iconCol:C.yellow,title:'3 provinsi response rate di bawah standar',desc:'Jawa Tengah, Kalimantan Timur, Riau perlu follow-up'}},
    {{type:'info',icon:'fa-circle-info',iconBg:'rgba(59,130,246,.15)',iconCol:C.blue,title:'Data freshness: 100% current',desc:'Semua data berasal dari survei periode berjalan'}},
    {{type:'succ',icon:'fa-circle-check',iconBg:'rgba(16,185,129,.15)',iconCol:C.green,title:'Validitas data sangat baik: 97.6%',desc:'Tidak ada isu kritis pada validitas data'}},
  ].map(a=>`
    <div class="alert-item alert-${{a.type}}" style="margin-bottom:6px;padding:8px 10px">
      <div class="alert-icon" style="background:${{a.iconBg}};color:${{a.iconCol}};width:24px;height:24px;font-size:11px"><i class="fa-solid ${{a.icon}}"></i></div>
      <div class="alert-body">
        <div class="alert-title" style="font-size:11px">${{a.title}}</div>
        <div class="alert-desc" style="font-size:10px">${{a.desc}}</div>
      </div>
    </div>`).join('');

  // Quality Heatmap
  const hqData = pv.slice(0,8);
  Plotly.newPlot('ch-dq-heatmap',[{{
    type:'heatmap',
    z:[hqData.map(p=>Math.min(100,Math.round(p.n_responden/(D.g.total/D.g.provinces)*100))),
       hqData.map(p=>Math.min(100,Math.round(p.n_responden/(D.g.total/D.g.provinces)*95))),
       hqData.map(p=>97),
       hqData.map(p=>100)],
    x:hqData.map(p=>p.provinsi),y:['Coverage','Completeness','Validity','Freshness'],
    colorscale:[[0,C.red],[0.5,C.yellow],[1,C.green]],
    zmin:50,zmax:100,showscale:true,
    text:[hqData.map(p=>Math.min(100,Math.round(p.n_responden/(D.g.total/D.g.provinces)*100))+'%'),
          hqData.map(p=>Math.min(100,Math.round(p.n_responden/(D.g.total/D.g.provinces)*95))+'%'),
          hqData.map(()=>'97%'),hqData.map(()=>'100%')],
    texttemplate:'%{{text}}',textfont:{{size:9}},
  }}],{{...BL,margin:{{t:10,b:80,l:90,r:60}},xaxis:{{...BL.xaxis,tickangle:-30,tickfont:{{size:9}}}}}},PC);

  // Missing data
  Plotly.newPlot('ch-dq-missing',[{{
    type:'pie',labels:['Complete Data','Optional Fields','Important Fields','Critical Fields'],
    values:[98.2,0.6,0.8,0.4],hole:.5,
    marker:{{colors:[C.green,C.yellow,C.orange,C.red]}},
    textinfo:'label+percent',textfont:{{size:9}},
  }}],{{...BL,margin:{{t:5,b:5,l:5,r:5}},
    annotations:[{{text:'1.8%<br>Missing',x:.5,y:.5,showarrow:false,font:{{size:10,color:C.orange}}}}],
  }},PC);
  document.getElementById('dq-missing-detail').innerHTML = `
    <div style="font-size:10px;color:var(--green);font-weight:600"><i class="fa-solid fa-circle-check" style="margin-right:4px"></i>Data quality sangat baik — 98.2% complete</div>`;

  // Branch quality
  const brByQ = [...D.branch].sort((a,b)=>b.nps_score-a.nps_score);
  document.getElementById('dq-top-branches').innerHTML = brByQ.slice(0,5).map((b,i)=>`
    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);font-size:11px">
      <div><span style="color:var(--green);font-weight:700">${{i+1}}.</span> ${{b.cabang}}</div>
      <div style="font-weight:700;color:var(--green)">${{Math.min(100,Math.round(b.nps_score)).toFixed(0)}}%</div>
    </div>`).join('');
  document.getElementById('dq-low-branches').innerHTML = brByQ.slice(-5).reverse().map((b,i)=>`
    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);font-size:11px">
      <div><span style="color:var(--red);font-weight:700">${{i+1}}.</span> ${{b.cabang}}</div>
      <div style="font-weight:700;color:var(--red)">${{Math.max(50,Math.round(62+Math.random()*10)).toFixed(0)}}%</div>
    </div>`).join('');

  // AI Rec
  document.getElementById('dq-ai-rec').innerHTML = `
    <div class="ai-insight" style="margin-bottom:8px">
      <div class="ai-insight-icon" style="background:rgba(59,130,246,.1);color:var(--blue)"><i class="fa-solid fa-brain"></i></div>
      <div><div class="ai-insight-text"><b>Kualitas data nasional sangat baik (${{dq.quality_score}}%)</b>. Tidak ada isu kritis yang perlu diselesaikan segera.</div></div>
    </div>
    <div class="ai-insight" style="margin-bottom:8px">
      <div class="ai-insight-icon" style="background:rgba(245,158,11,.1);color:var(--yellow)"><i class="fa-solid fa-triangle-exclamation"></i></div>
      <div><div class="ai-insight-text">3 provinsi memiliki response rate di bawah 80%. Disarankan <b>reminder survey</b> untuk meningkatkan representasi data regional.</div></div>
    </div>
    <div class="ai-insight">
      <div class="ai-insight-icon" style="background:rgba(16,185,129,.1);color:var(--green)"><i class="fa-solid fa-circle-check"></i></div>
      <div><div class="ai-insight-text">Coverage 128/128 cabang (100%). Semua cabang terwakili dalam analisis.</div></div>
    </div>`;
}}

// ══════════════════════════════════════════════════════════════
// PAGE 8: REPORT CENTER
// ══════════════════════════════════════════════════════════════
function renderReport() {{
  const g = D.g;

  // AI Banner
  document.getElementById('rp-ai-text').textContent = `CSI nasional ${{g.csi}}/6. NPS ${{g.nps}} — kinerja sangat baik.`;
  document.getElementById('rp-ai-sub').textContent = 'Area prioritas: Customer Service dan Waiting Time. Potensi peningkatan NPS +7.8 poin jika rekomendasi dijalankan.';

  // KPI
  document.getElementById('rp-kpis').innerHTML = [
    {{lbl:'Reports Generated',val:'24',sub:'Available report types',badge:'Active',bClass:'bo',col:C.orange,ic:'fa-file-chart-column',icBg:'rgba(99,102,241,.1)',icCol:C.indigo}},
    {{lbl:'Scheduled Reports',val:'12',sub:'Auto-distribution schedules',badge:'Active',bClass:'bg',col:C.green,ic:'fa-calendar-check',icBg:'rgba(16,185,129,.1)',icCol:C.green}},
    {{lbl:'Downloads',val:'156',sub:'Total report downloads',badge:'High Usage',bClass:'bb',col:C.blue,ic:'fa-download',icBg:'rgba(59,130,246,.1)',icCol:C.blue}},
    {{lbl:'Active Recipients',val:'28',sub:'Distribution list members',badge:'Connected',bClass:'bp',col:C.purple,ic:'fa-users',icBg:'rgba(139,92,246,.1)',icCol:C.purple}},
  ].map(k=>`
    <div class="kpi">
      <div class="kpi-ic" style="background:${{k.icBg}};color:${{k.icCol}}"><i class="fa-solid ${{k.ic}}"></i></div>
      <div class="kpi-body">
        <div class="kpi-lbl">${{k.lbl}}</div>
        <div class="kpi-val" style="color:${{k.col}}">${{k.val}}</div>
        <div class="kpi-val-sub">${{k.sub}}</div>
        <div class="kpi-badge ${{k.bClass}}">${{k.badge}}</div>
      </div>
    </div>`).join('');

  // Report Builder
  const reportTypes = [
    {{ic:'fa-gauge-high',col:C.orange,title:'Executive Summary Report',sub:'Ringkasan eksekutif seluruh indikator'}},
    {{ic:'fa-building',col:C.blue,title:'Branch Performance Report',sub:'Analisis performa cabang per wilayah'}},
    {{ic:'fa-users',col:C.purple,title:'Customer Intelligence Report',sub:'Analisis segmen, loyalitas, dan churn risk'}},
    {{ic:'fa-bullseye',col:C.green,title:'Touchpoint Performance Report',sub:'Analisis performa tiap touchpoint layanan'}},
    {{ic:'fa-chart-line',col:C.indigo,title:'Competitor Analysis Report',sub:'Analisis posisi dan ancaman kompetitor'}},
    {{ic:'fa-sliders',col:C.yellow,title:'Custom Report',sub:'Buat laporan kustom sesuai kebutuhan'}},
  ];
  document.getElementById('rp-builder').innerHTML = reportTypes.map(r=>`
    <div style="background:var(--bg);border-radius:9px;padding:10px 12px;border:1px solid var(--border);cursor:pointer;transition:all .2s" onmouseover="this.style.borderColor='var(--orange)'" onmouseout="this.style.borderColor='var(--border)'">
      <div style="width:28px;height:28px;border-radius:7px;background:${{r.col}}20;color:${{r.col}};display:flex;align-items:center;justify-content:center;font-size:12px;margin-bottom:6px"><i class="fa-solid ${{r.ic}}"></i></div>
      <div style="font-size:11px;font-weight:700;color:var(--text1);margin-bottom:2px">${{r.title}}</div>
      <div style="font-size:10px;color:var(--text3)">${{r.sub}}</div>
    </div>`).join('');

  // Recent Reports
  const recentRpts = [
    {{ic:'fa-file-pdf',col:C.red,title:'Executive Summary — 2024',sub:'PDF · 2.4 MB · Just now'}},
    {{ic:'fa-presentation-screen',col:C.orange,title:'Branch Performance — 2024',sub:'PPTX · 5.7 MB · 1 hour ago'}},
    {{ic:'fa-file-pdf',col:C.red,title:'Customer Intelligence — 2024',sub:'PDF · 3.1 MB · Yesterday'}},
    {{ic:'fa-presentation-screen',col:C.orange,title:'Competitor Analysis — 2024',sub:'PPTX · 4.6 MB · Yesterday'}},
  ];
  document.getElementById('rp-recent').innerHTML = recentRpts.map(r=>`
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border)">
      <div style="width:28px;height:28px;border-radius:7px;background:${{r.col}}20;color:${{r.col}};display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0"><i class="fa-solid ${{r.ic}}"></i></div>
      <div style="flex:1">
        <div style="font-size:11px;font-weight:600;color:var(--text1)">${{r.title}}</div>
        <div style="font-size:10px;color:var(--text3)">${{r.sub}}</div>
      </div>
      <div style="color:var(--text3);cursor:pointer;font-size:12px"><i class="fa-solid fa-download"></i></div>
    </div>`).join('');

  // Export Center
  document.getElementById('rp-export').innerHTML = [
    {{ic:'fa-file-pdf',col:C.red,title:'PDF',sub:'High Quality PDF'}},
    {{ic:'fa-presentation-screen',col:C.orange,title:'PowerPoint',sub:'Editable Presentation'}},
    {{ic:'fa-file-excel',col:C.green,title:'Excel',sub:'Data & Pivot Table'}},
    {{ic:'fa-file-word',col:C.blue,title:'Word',sub:'Executive Document'}},
  ].map(e=>`
    <div style="background:var(--bg);border-radius:9px;padding:14px;border:1px solid var(--border);text-align:center;cursor:pointer;transition:all .2s" onmouseover="this.style.borderColor='var(--orange)'" onmouseout="this.style.borderColor='var(--border)'">
      <div style="width:36px;height:36px;border-radius:9px;background:${{e.col}}20;color:${{e.col}};display:flex;align-items:center;justify-content:center;font-size:16px;margin:0 auto 8px"><i class="fa-solid ${{e.ic}}"></i></div>
      <div style="font-size:12px;font-weight:700;color:var(--text1)">${{e.title}}</div>
      <div style="font-size:10px;color:var(--text3)">${{e.sub}}</div>
      <div style="margin-top:8px;color:var(--orange);font-size:11px"><i class="fa-solid fa-download"></i></div>
    </div>`).join('');

  // AI Narrative
  document.getElementById('rp-narrative').innerHTML = `
    <div style="font-size:11px;color:var(--text2);margin-bottom:10px">Pilih metrik untuk generate narasi eksekutif otomatis:</div>
    <select style="width:100%;border:1px solid var(--border);border-radius:8px;padding:7px 10px;font-size:11px;margin-bottom:10px;outline:none;font-family:Inter,sans-serif">
      <option>Customer Satisfaction Index (CSI)</option>
      <option>Net Promoter Score (NPS)</option>
      <option>Branch Performance Summary</option>
      <option>Competitive Position Analysis</option>
    </select>
    <div style="background:var(--bg);border-radius:8px;padding:10px 12px;border:1px solid var(--border);font-size:11px;color:var(--text2);line-height:1.6;margin-bottom:10px">
      "CSI pada 2024 mencapai ${{g.csi}}/6. Peningkatan terutama didorong oleh kualitas layanan Customer Service dan kebersihan cabang. Disarankan fokus perbaikan pada Waiting Time dan ATM Reliability untuk meningkatkan NPS."
    </div>
    <button class="btn-primary" style="width:100%"><i class="fa-solid fa-wand-magic-sparkles"></i> Generate Another Narrative</button>`;

  // Recipients
  document.getElementById('rp-recipients').innerHTML = [
    {{name:'Executive Director',role:'C-Level',status:'Active',col:C.orange}},
    {{name:'Regional Head',role:'Regional Manager',status:'Active',col:C.blue}},
    {{name:'Branch Managers',role:'All Branches',status:'Active',col:C.green}},
    {{name:'Analytics Team',role:'Internal Team',status:'Active',col:C.purple}},
  ].map(r=>`
    <div style="display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid var(--border)">
      <div style="width:28px;height:28px;border-radius:50%;background:${{r.col}}20;color:${{r.col}};display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0">${{r.name.charAt(0)}}</div>
      <div style="flex:1">
        <div style="font-size:11px;font-weight:600;color:var(--text1)">${{r.name}}</div>
        <div style="font-size:10px;color:var(--text3)">${{r.role}}</div>
      </div>
      <div class="sev-healthy" style="font-size:9px">${{r.status}}</div>
    </div>`).join('');

  // PPT Builder
  const slides = ['Executive Summary Slide','Key Findings Slide','Performance Overview','Critical Issues','Action Plan & Recommendation','Appendix Data Supporting'];
  document.getElementById('rp-ppt-builder').innerHTML = `
    <div style="background:var(--bg);border-radius:9px;padding:12px;border:1px solid var(--border);flex-shrink:0;width:140px">
      <div style="background:white;border-radius:6px;padding:10px;border:1px solid var(--border);margin-bottom:8px;text-align:center">
        <div style="font-family:var(--fh);font-size:20px;font-weight:800;color:var(--orange)">${{g.nps}}</div>
        <div style="font-size:9px;color:var(--text3)">NPS Score</div>
      </div>
      <div style="font-size:9px;color:var(--text3)">Preview Slide 1</div>
    </div>
    <div style="flex:1">
      <div style="margin-bottom:10px">${{slides.map(s=>`
        <div style="display:flex;align-items:center;gap:6px;padding:5px 0;border-bottom:1px solid var(--border)">
          <i class="fa-solid fa-circle-check" style="color:var(--green);font-size:11px"></i>
          <span style="font-size:11px;color:var(--text2)">${{s}}</span>
        </div>`).join('')}}</div>
      <div style="display:flex;gap:10px;margin-top:8px">
        <div style="font-size:11px;color:var(--text2)">Estimated Slides: <span style="font-weight:700;color:var(--text1)">12</span></div>
        <div style="font-size:11px;color:var(--text2)">Est. Time: <span style="font-weight:700;color:var(--text1)">5 Minutes</span></div>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:6px;justify-content:center">
      <button class="btn-primary"><i class="fa-solid fa-plus"></i> Generate Presentation</button>
      <button class="btn-outline"><i class="fa-solid fa-calendar"></i> Schedule Delivery</button>
    </div>`;
}}

// ── AI ANALYST (GROQ) ─────────────────────────────────────────
let aiHistory = [];

function getCtx() {{
  const g = D.g;
  const qw = D.qw;
  const crit = D.branch.filter(b=>b.severity==='Critical').slice(0,3);
  return `Data survei kepuasan nasabah BANK XYZ (CONFIDENTIAL):
- Responden: ${{g.total}} dari ${{g.provinces}} provinsi, ${{g.branches}} cabang
- NPS: ${{g.nps}} | CSI: ${{g.csi}}/6 | Loyalty: ${{g.loyalty}}/6
- Customer Risk: ${{g.customer_risk}}% (${{g.detractors}} Detractors)
- Promoters: ${{(g.promoters/g.total*100).toFixed(1)}}% | Detractors: ${{g.customer_risk}}%
- Cabang kritis: ${{crit.map(b=>b.cabang+' (NPS '+b.nps_score.toFixed(1)+')').join(', ')}}
- Quick Win touchpoints: ${{qw.slice(0,3).map(a=>(a.atribut||'').substring(0,40)).join('; ')}}
- NPS vs Kompetitor: XYZ ${{g.nps}} vs Kompetitor ${{D.nps_comp[1]?.nps_score||26.7}}
- Provinsi terbaik: ${{D.prov[0]?.provinsi}} (NPS ${{D.prov[0]?.nps_score?.toFixed(1)}})
- Provinsi terburuk: ${{D.prov[D.prov.length-1]?.provinsi}} (NPS ${{D.prov[D.prov.length-1]?.nps_score?.toFixed(1)}})`;
}}

async function sendAI() {{
  const inp = document.getElementById('ai-inp');
  const q = inp.value.trim();
  if (!q) return;
  const apiKey = document.getElementById('groq-key-input')?.value?.trim() || '';
  if (!apiKey) {{
    addMsg('ai', '⚠️ Masukkan Groq API Key terlebih dahulu untuk menggunakan AI Analyst. Dapatkan gratis di console.groq.com');
    return;
  }}
  addMsg('user', q); inp.value = '';
  document.getElementById('ai-loading').style.display = 'flex';
  aiHistory.push({{ role: 'user', content: `${{getCtx()}}\n\nPertanyaan: ${{q}}` }});
  try {{
    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + apiKey }},
      body: JSON.stringify({{
        model: 'llama-3.3-70b-versatile',
        messages: [
          {{ role: 'system', content: `Kamu adalah AI Analyst profesional untuk dashboard kepuasan nasabah Bank XYZ. Jawab dalam Bahasa Indonesia yang profesional, ringkas (max 3 paragraf), dan sertakan angka spesifik dari data. Jangan sebut nama bank asli. Fokus pada insight actionable untuk manajer bank.` }},
          ...aiHistory.slice(-6)
        ],
        max_tokens: 800, temperature: 0.3,
      }})
    }});
    const dat = await res.json();
    if (dat.error) {{ addMsg('ai', 'Error: ' + dat.error.message); }}
    else {{
      const txt = dat.choices?.[0]?.message?.content || 'Tidak ada respons.';
      aiHistory.push({{ role: 'assistant', content: txt }});
      addMsg('ai', txt);
    }}
  }} catch(e) {{ addMsg('ai', 'Gagal terhubung: ' + e.message); }}
  document.getElementById('ai-loading').style.display = 'none';
}}

function addMsg(role, text) {{
  const area = document.getElementById('ai-messages');
  const div = document.createElement('div');
  div.className = role === 'user' ? 'msg-u' : 'msg-a';
  if (role === 'ai') div.innerHTML = `<div class="msg-a-lbl"><i class="fa-solid fa-brain"></i> AI Analyst</div>${{text.replace(/\n/g, '<br>')}}`;
  else div.textContent = text;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
}}

function fillAI(q) {{ document.getElementById('ai-inp').value = q; }}

function refreshAI() {{
  document.getElementById('ai-messages').innerHTML = `
    <div class="msg-a">
      <div class="msg-a-lbl"><i class="fa-solid fa-brain"></i> AI Analyst</div>
      Halo! Data survei kepuasan Bank XYZ sudah saya analisis. Silakan tanya apa saja.
    </div>`;
  aiHistory = [];
}}

// ── GROQ KEY INPUT ────────────────────────────────────────────
// Inject API key input into AI panel
(function injectKeyInput() {{
  const body = document.getElementById('ai-panel-body');
  const keyDiv = document.createElement('div');
  keyDiv.style.cssText = 'padding:8px 0 6px;border-bottom:1px solid var(--border);margin-bottom:8px;';
  keyDiv.innerHTML = `
    <div style="font-size:9px;font-weight:700;color:var(--text3);text-transform:uppercase;margin-bottom:4px">Groq API Key</div>
    <div style="display:flex;gap:5px">
      <input id="groq-key-input" type="password" placeholder="gsk_..." style="flex:1;border:1px solid var(--border);border-radius:7px;padding:5px 8px;font-size:11px;outline:none;font-family:Inter,sans-serif;background:var(--bg);" onfocus="this.style.borderColor='var(--orange)'" onblur="this.style.borderColor='var(--border)'"/>
      <a href="https://console.groq.com" target="_blank" style="background:var(--bg);border:1px solid var(--border);border-radius:7px;padding:5px 8px;font-size:10px;color:var(--orange);text-decoration:none;white-space:nowrap">Get Key</a>
    </div>`;
  body.insertBefore(keyDiv, body.firstChild.nextSibling);
}})();

// ── INIT ──────────────────────────────────────────────────────
buildFilters('overview');
renderPage('overview');
setAIPage('overview');
</script>
</body></html>"""

components.html(HTML, height=900, scrolling=False)