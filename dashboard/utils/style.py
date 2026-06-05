import streamlit as st
import pandas as pd
import re

# ── Color Palette ──────────────────────────────────────────────
PRIMARY        = "#C8410B"
PRIMARY_DARK   = "#A33509"
PRIMARY_LIGHT  = "#F4845F"
BG_SIDEBAR_TOP = "#4A1E0E"
BG_SIDEBAR_BOT = "#1C0A04"

# ── Dark Mode Helpers ──────────────────────────────────────────
def is_dark_mode():
    return st.session_state.get('dark_mode', False)

def get_theme():
    """Return color tokens based on current light/dark mode."""
    if st.session_state.get('dark_mode', False):
        return dict(
            BG_MAIN        = "#0F1729",
            BG_CARD        = "#1A2540",
            TEXT_DARK      = "#E2E8F0",
            TEXT_SECONDARY = "#94A3B8",
            TEXT_MUTED     = "#64748B",
            TEXT_LIGHT     = "#F1F5F9",
            BORDER         = "#2D3F5F",
        )
    return dict(
        BG_MAIN        = "#F0F2F7",
        BG_CARD        = "#FFFFFF",
        TEXT_DARK      = "#111827",
        TEXT_SECONDARY = "#374151",
        TEXT_MUTED     = "#4B5563",
        TEXT_LIGHT     = "#F9FAFB",
        BORDER         = "#E5E7EB",
    )

# ── Resolve tokens at import time (re-resolved inside functions) ─
BG_MAIN        = "#F0F2F7"
BG_CARD        = "#FFFFFF"
TEXT_DARK      = "#111827"
TEXT_SECONDARY = "#374151"
TEXT_MUTED     = "#4B5563"
TEXT_LIGHT     = "#F9FAFB"
BORDER         = "#E5E7EB"

COLOR_GREEN    = "#22c55e"
COLOR_YELLOW   = "#f59e0b"
COLOR_ORANGE   = "#f97316"
COLOR_RED      = "#ef4444"
COLOR_BLUE     = "#3b82f6"
COLOR_PURPLE   = "#8b5cf6"
COLOR_TEAL     = "#14b8a6"

# ── NPS Category Colors (4-tier) ───────────────────────────────
NPS_COLOR_90   = "#22c55e"   # 90–100 : hijau terang
NPS_COLOR_75   = "#84cc16"   # 75–89  : hijau lime / keruh
NPS_COLOR_60   = "#f97316"   # 60–74  : oranye
NPS_COLOR_LOW  = "#ef4444"   # < 60   : merah
NPS_COLOR_NA   = "#94a3b8"   # no data: abu

CHART_COLORS = [PRIMARY, "#3B82F6", "#22C55E", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6"]

NPS_TARGET      = 50
CSI_TARGET      = 75
LOYALTY_TARGET  = 70
RISK_THRESHOLD  = 20
TOTAL_BRANCHES  = 128

# ── GeoJSON Mapping ────────────────────────────────────────────
GEOJSON_TO_STANDARD = {
    "Aceh":"Aceh","Bali":"Bali","BangkaBelitung":"Bangka Belitung",
    "Banten":"Banten","Bengkulu":"Bengkulu","Gorontalo":"Gorontalo",
    "JakartaRaya":"DKI Jakarta","Jambi":"Jambi","JawaBarat":"Jawa Barat",
    "JawaTengah":"Jawa Tengah","JawaTimur":"Jawa Timur",
    "KalimantanBarat":"Kalimantan Barat","KalimantanSelatan":"Kalimantan Selatan",
    "KalimantanTengah":"Kalimantan Tengah","KalimantanTimur":"Kalimantan Timur",
    "KalimantanUtara":"Kalimantan Utara","KepulauanRiau":"Kepulauan Riau",
    "Lampung":"Lampung","Maluku":"Maluku","MalukuUtara":"Maluku Utara",
    "NusaTenggaraBarat":"Nusa Tenggara Barat","NusaTenggaraTimur":"Nusa Tenggara Timur",
    "Papua":"Papua","PapuaBarat":"Papua Barat","Riau":"Riau",
    "SulawesiBarat":"Sulawesi Barat","SulawesiSelatan":"Sulawesi Selatan",
    "SulawesiTengah":"Sulawesi Tengah","SulawesiTenggara":"Sulawesi Tenggara",
    "SulawesiUtara":"Sulawesi Utara","SumateraBarat":"Sumatera Barat",
    "SumateraSelatan":"Sumatera Selatan","SumateraUtara":"Sumatera Utara",
    "Yogyakarta":"DI Yogyakarta",
}
STANDARD_TO_GEOJSON = {v: k for k, v in GEOJSON_TO_STANDARD.items()}

def normalize_prov_name(name):
    return re.sub(r'[\s\-_./]', '', str(name)).lower()

def match_prov_to_geojson(prov_name):
    if prov_name in STANDARD_TO_GEOJSON:
        return STANDARD_TO_GEOJSON[prov_name]
    norm = normalize_prov_name(prov_name)
    for g, s in GEOJSON_TO_STANDARD.items():
        if normalize_prov_name(g) == norm or normalize_prov_name(s) == norm:
            return g
    for g, s in GEOJSON_TO_STANDARD.items():
        if norm in normalize_prov_name(s) or normalize_prov_name(s) in norm:
            return g
    return prov_name

def prepare_prov_for_map(prov_df, prov_col='PROV'):
    df = prov_df.copy()
    df['GEOJSON_NAME'] = df[prov_col].apply(match_prov_to_geojson)
    return df

def set_page_config(title, icon="📊"):
    st.set_page_config(
        page_title=f"{title} — BankSurvey",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ── SVG Helpers ────────────────────────────────────────────────
def _svg(path_d, w=14, h=14):
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round">{path_d}</svg>')

SVG_MAP        = _svg('<polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/>')
SVG_BRANCH     = _svg('<line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>')
SVG_ANALYTICS  = _svg('<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/>')
SVG_TARGET     = _svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>')
SVG_WARNING    = _svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>')
SVG_CHECK      = _svg('<polyline points="20 6 9 17 4 12"/>')
SVG_SHIELD     = _svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>')
SVG_USERS      = _svg('<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>')
SVG_STAR       = _svg('<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>')
SVG_HEART      = _svg('<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>')
SVG_RISK       = _svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>')
SVG_TREND_UP   = _svg('<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>')
SVG_TREND_DOWN = _svg('<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>')
SVG_BOLT       = _svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>')
SVG_INFO       = _svg('<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>')
SVG_VERIFIED   = _svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>')
SVG_REPORT     = _svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>')
SVG_TEAL       = _svg('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>')


def inject_global_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif !important;
        color: {TEXT_DARK};
    }}

    /* ══════════════════════════════════════════════
       PAGE TITLE & SUBTITLE
    ══════════════════════════════════════════════ */
    .page-title {{
        font-size: 20px;
        font-weight: 700;
        color: {TEXT_DARK};
        line-height: 1.2;
        margin-bottom: 2px;
    }}
    .page-subtitle {{
        font-size: 12px;
        color: {TEXT_MUTED};
        margin-bottom: 8px;
    }}

    /* ══════════════════════════════════════════════
       WHITE CARD SYSTEM — MULTI-SELECTOR ROBUST FIX
       Covers Streamlit 1.28–1.40+ (testid changes)
    ══════════════════════════════════════════════ */

    /* Outer wrapper: strip any default styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }}

    /* Inner div: apply white card */
    div[data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
        padding: 12px 14px !important;
        overflow: visible !important;
    }}

    /* Prevent double-border on nested containers */
    div[data-testid="stVerticalBlockBorderWrapper"] > div
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        border-radius: 0 !important;
    }}

    /* ── Extra: ensure stVerticalBlock inside border wrapper is white ── */
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {{
        background: {BG_CARD} !important;
    }}

    /* ── Pastikan nested stVerticalBlock tidak double border ── */
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"]
    div[data-testid="stVerticalBlock"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    /* ── Utility: manual .dash-card class for inline HTML cards ── */
    .dash-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }}

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header {{ visibility: hidden; height: 0; }}
    .stDeployButton {{ display: none !important; }}
    [data-testid="stSidebarNav"] {{ display: none !important; }}
    .stApp {{ background-color: {BG_MAIN}; }}

    /* ══════════════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════════════ */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BG_SIDEBAR_TOP} 0%, {BG_SIDEBAR_BOT} 100%) !important;
        border-right: none !important;
        min-width: 200px !important;
        max-width: 200px !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{ padding-top: 0 !important; }}
    [data-testid="stSidebar"] [data-testid="stPageLink"] a {{
        color: #C4B5A5 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 6px 10px !important;
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        text-decoration: none !important;
        transition: all 0.15s ease !important;
        margin: 1px 6px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{
        background: rgba(200,65,11,0.2) !important;
        color: #F4845F !important;
    }}
    [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {{
        background: {PRIMARY} !important;
        color: white !important;
        font-weight: 700 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stPageLink"] p {{
        font-size: 0.8rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    [data-testid="stSidebarCollapseButton"] {{
        background: {PRIMARY} !important;
        border-radius: 50% !important;
    }}

    /* ══════════════════════════════════════════════
       MAIN CONTAINER — COMPACT
    ══════════════════════════════════════════════ */
    .main .block-container {{
        padding: 0.5rem 0.9rem 0.5rem 0.9rem !important;
        max-width: 100% !important;
    }}
    .element-container {{ margin-bottom: 0.2rem !important; }}
    .stMarkdown {{ margin-bottom: 0 !important; }}
    [data-testid="stHorizontalBlock"] {{ gap: 10px !important; }}
    .stVerticalBlock {{ gap: 0.3rem !important; }}
    section[data-testid="stSidebar"] .stVerticalBlock {{ gap: 0.02rem !important; }}
    div[data-testid="column"] > div > div[data-testid="stVerticalBlock"] {{
        gap: 0.3rem !important;
    }}

    /* ══════════════════════════════════════════════
       SELECTBOX
    ══════════════════════════════════════════════ */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background: white !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        min-height: 36px !important;
    }}
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    [data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p {{
        color: {TEXT_DARK} !important;
        font-size: 0.8rem !important;
    }}
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] [role="option"] {{
        color: {TEXT_DARK} !important;
        font-size: 0.8rem !important;
        background: white !important;
    }}
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [role="option"]:hover {{
        background: #FFF7ED !important;
        color: {PRIMARY} !important;
    }}
    [data-testid="stSelectbox"] label {{
        font-size: 9px !important;
        font-weight: 700 !important;
        color: {TEXT_MUTED} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }}

    /* ══════════════════════════════════════════════
       KPI CARD
    ══════════════════════════════════════════════ */
    .kpi-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        padding: 10px 12px;
        min-height: 110px;
        height: auto;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
        box-sizing: border-box;
    }}
    .kpi-icon-box {{
        width: 28px; height: 28px;
        border-radius: 7px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        margin-bottom: 2px;
    }}
    .kpi-label {{
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: {TEXT_MUTED};
    }}
    .kpi-value {{
        font-size: 26px;
        font-weight: 700;
        color: {TEXT_DARK};
        line-height: 1.0;
        letter-spacing: -0.02em;
    }}
    .kpi-value-sm {{
        font-size: 20px;
        font-weight: 700;
        color: {TEXT_DARK};
        line-height: 1.0;
    }}
    .kpi-badge {{
        display: inline-flex;
        align-items: center;
        font-size: 9px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 20px;
    }}
    .kpi-badge.green  {{ background:#dcfce7; color:#166534; }}
    .kpi-badge.yellow {{ background:#fef9c3; color:#713f12; }}
    .kpi-badge.red    {{ background:#fee2e2; color:#991b1b; }}
    .kpi-badge.blue   {{ background:#dbeafe; color:#1e40af; }}
    .kpi-badge.orange {{ background:#ffedd5; color:#9a3412; }}
    .kpi-badge.teal   {{ background:#ccfbf1; color:#0f766e; }}

    /* ══════════════════════════════════════════════
       ALERT CARD
    ══════════════════════════════════════════════ */
    .alert-card {{
        border-radius: 10px;
        padding: 8px 11px;
        display: flex; align-items: center; gap: 8px;
        border: 1px solid transparent;
        min-height: 62px; height: auto; box-sizing: border-box;
    }}
    .alert-card.critical {{ background:#fff1f2; border-color:#fecdd3; }}
    .alert-card.warning  {{ background:#fffbeb; border-color:#fde68a; }}
    .alert-card.success  {{ background:#f0fdf4; border-color:#bbf7d0; }}
    .alert-card.info     {{ background:#eff6ff; border-color:#bfdbfe; }}
    .alert-icon {{
        width:28px; height:28px; border-radius:50%;
        display:flex; align-items:center; justify-content:center; flex-shrink:0;
    }}
    .alert-card.critical .alert-icon {{ background:#fee2e2; }}
    .alert-card.warning  .alert-icon {{ background:#fef3c7; }}
    .alert-card.success  .alert-icon {{ background:#dcfce7; }}
    .alert-card.info     .alert-icon {{ background:#dbeafe; }}
    .alert-card-body {{ flex:1; min-width:0; }}
    .alert-card-label {{
        font-size:9px; font-weight:700;
        text-transform:uppercase; letter-spacing:0.08em;
    }}
    .alert-card.critical .alert-card-label {{ color:#be123c; }}
    .alert-card.warning  .alert-card-label {{ color:#92400e; }}
    .alert-card.success  .alert-card-label {{ color:#166534; }}
    .alert-card.info     .alert-card-label {{ color:#1e40af; }}
    .alert-card-value {{
        font-size:13px; font-weight:700; line-height:1.1;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    }}
    .alert-card.critical .alert-card-value {{ color:#be123c; }}
    .alert-card.warning  .alert-card-value {{ color:#92400e; }}
    .alert-card.success  .alert-card-value {{ color:#166534; }}
    .alert-card.info     .alert-card-value {{ color:#1e40af; }}
    .alert-card-sub {{
        font-size:9.5px; color:{TEXT_MUTED};
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    }}

    /* ══════════════════════════════════════════════
       SECTION HEADER
    ══════════════════════════════════════════════ */
    .sec-hdr {{
        display:flex; align-items:center; gap:6px;
        font-size:11.5px; font-weight:700; color:{TEXT_DARK};
        margin-bottom:8px;
    }}
    .sec-hdr-icon {{
        width:22px; height:22px; border-radius:6px;
        display:flex; align-items:center; justify-content:center; flex-shrink:0;
    }}

    /* ══════════════════════════════════════════════
       GLOBAL HEADER
    ══════════════════════════════════════════════ */
    .g-header {{
        display:flex; align-items:center; justify-content:space-between;
        background:{BG_CARD}; border:1px solid {BORDER}; border-radius:10px;
        padding:7px 14px; margin-bottom:10px;
        box-shadow:0 1px 4px rgba(0,0,0,0.04);
    }}
    .g-header-right {{ display:flex; align-items:center; gap:6px; }}
    .h-btn {{
        display:inline-flex; align-items:center; gap:4px;
        padding:4px 9px; border-radius:6px; border:1px solid {BORDER};
        background:white; font-size:10.5px; font-weight:600;
        color:{TEXT_SECONDARY}; cursor:pointer;
    }}
    .h-btn.primary {{
        background:{PRIMARY}; color:white; border-color:{PRIMARY};
    }}
    .h-ts {{
        font-size:9.5px; color:{TEXT_MUTED};
        display:flex; align-items:center; gap:3px;
    }}

    /* ══════════════════════════════════════════════
       AI BANNER
    ══════════════════════════════════════════════ */
    .ai-banner {{
        background:linear-gradient(135deg,#fff7ed,#fff);
        border:1px solid #fed7aa; border-left:3px solid {PRIMARY};
        border-radius:9px; padding:8px 11px; margin-top:6px; margin-bottom:10px;
        display:flex; align-items:flex-start; gap:8px;
    }}
    .ai-banner-icon {{
        width:26px; height:26px;
        background:linear-gradient(135deg,{PRIMARY},{PRIMARY_LIGHT});
        border-radius:7px; display:flex; align-items:center;
        justify-content:center; flex-shrink:0;
        font-weight:800; font-size:0.65rem; color:white;
    }}
    .ai-label {{ font-size:9px; font-weight:700;
                text-transform:uppercase; letter-spacing:0.08em; color:{PRIMARY}; }}
    .ai-text  {{ font-size:11.5px; font-weight:600; color:{TEXT_DARK}; }}
    .ai-sub   {{ font-size:10.5px; color:{TEXT_MUTED}; margin-top:1px; }}

    /* ══════════════════════════════════════════════
       CHAT
    ══════════════════════════════════════════════ */
    .chat-box {{
        background:#F9FAFB; border:1px solid {BORDER}; border-radius:9px;
        padding:7px; max-height:200px; min-height:60px; overflow-y:auto;
        display:flex; flex-direction:column; gap:5px; margin-bottom:6px;
    }}
    .chat-user {{
        background:{PRIMARY}; color:white;
        border-radius:9px 9px 3px 9px;
        padding:5px 9px; font-size:10.5px;
        align-self:flex-end; max-width:88%;
    }}
    .chat-ai {{
        background:white; color:{TEXT_DARK};
        border:1px solid {BORDER};
        border-radius:9px 9px 9px 3px;
        padding:5px 9px; font-size:10.5px;
        align-self:flex-start; max-width:92%;
    }}

    /* ══════════════════════════════════════════════
       TABLES
    ══════════════════════════════════════════════ */
    .styled-table {{ width:100%; border-collapse:collapse; font-size:10.5px; }}
    .styled-table th {{
        background:#F9FAFB; color:{TEXT_MUTED}; font-weight:700;
        font-size:9px; text-transform:uppercase; letter-spacing:0.05em;
        padding:5px 7px; border-bottom:2px solid {BORDER}; text-align:left;
        position:sticky; top:0; z-index:1;
    }}
    .styled-table td {{
        padding:4px 7px; border-bottom:1px solid #F3F4F6; color:{TEXT_DARK};
    }}
    .styled-table tr:hover td {{ background:#FAFAFA; }}

    /* ══════════════════════════════════════════════
       BADGES
    ══════════════════════════════════════════════ */
    .badge {{ display:inline-flex; align-items:center;
              font-size:9px; font-weight:700; padding:2px 6px; border-radius:20px; }}
    .badge-green  {{ background:#dcfce7; color:#166534; }}
    .badge-yellow {{ background:#fef9c3; color:#713f12; }}
    .badge-red    {{ background:#fee2e2; color:#991b1b; }}
    .badge-blue   {{ background:#dbeafe; color:#1e40af; }}
    .badge-orange {{ background:#ffedd5; color:#9a3412; }}

    /* ══════════════════════════════════════════════
       TABS
    ══════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {{
        background:transparent; border-bottom:2px solid {BORDER}; gap:0;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size:11px; font-weight:500; color:{TEXT_MUTED};
        padding:4px 12px; border-bottom:2px solid transparent; margin-bottom:-2px;
    }}
    .stTabs [aria-selected="true"] {{
        color:{PRIMARY} !important;
        border-bottom:2px solid {PRIMARY} !important;
        font-weight:700;
    }}

    /* ══════════════════════════════════════════════
       BUTTONS
    ══════════════════════════════════════════════ */
    .stButton > button {{
        font-size:10px !important; padding:3px 8px !important;
        border-radius:6px !important; border:1px solid {BORDER} !important;
        background:white !important; color:{TEXT_DARK} !important;
    }}
    .stButton > button:hover {{
        background:{PRIMARY} !important; color:white !important;
        border-color:{PRIMARY} !important;
    }}

    /* ══════════════════════════════════════════════
       MISC
    ══════════════════════════════════════════════ */
    [data-testid="stPlotlyChart"] {{ border-radius:6px; overflow:hidden; }}
    ::-webkit-scrollbar {{ width:3px; height:3px; }}
    ::-webkit-scrollbar-thumb {{ background:#D1D5DB; border-radius:10px; }}
    hr {{ border:none; border-top:1px solid {BORDER}; margin:6px 0; }}

    /* ══════════════════════════════════════════════
       AI COPILOT PANEL
    ══════════════════════════════════════════════ */
    .ai-copilot-header {{
        background: linear-gradient(135deg, {PRIMARY}, {PRIMARY_LIGHT});
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 10px;
    }}
    .ai-insight-box {{
        background: #FFF7ED;
        border: 1px solid #FED7AA;
        border-left: 3px solid {PRIMARY};
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 10px;
        font-size: 10.5px;
        font-weight: 600;
        color: #0F172A;
        line-height: 1.4;
    }}
    .ai-section-label {{
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: {TEXT_MUTED};
        margin-bottom: 6px;
    }}
    .ai-finding-row {{
        display: flex;
        gap: 8px;
        padding: 5px 0;
        border-bottom: 1px solid #F3F4F6;
        align-items: flex-start;
    }}
    .ai-chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-bottom: 8px;
    }}
    .ai-chip {{
        background: #F1F5F9;
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 9.5px;
        font-weight: 600;
        color: #334155;
        cursor: pointer;
        white-space: nowrap;
    }}
    .ai-chip:hover {{
        background: #FFF7ED;
        border-color: {PRIMARY};
        color: {PRIMARY};
    }}

    /* ══════════════════════════════════════════════
       EXECUTIVE CARD ACCENT
    ══════════════════════════════════════════════ */
    .exec-card {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .exec-card-accent-red    {{ border-top: 3px solid {COLOR_RED}; }}
    .exec-card-accent-green  {{ border-top: 3px solid {COLOR_GREEN}; }}
    .exec-card-accent-blue   {{ border-top: 3px solid {COLOR_BLUE}; }}
    .exec-card-accent-orange {{ border-top: 3px solid {COLOR_ORANGE}; }}
    .exec-card-accent-primary {{ border-top: 3px solid {PRIMARY}; }}

    /* ══════════════════════════════════════════════
       PROGRESS BAR
    ══════════════════════════════════════════════ */
    .prog-bar-wrap {{
        background: #F3F4F6;
        border-radius: 6px;
        height: 6px;
        overflow: hidden;
    }}
    .prog-bar-fill {{
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }}

    /* ══════════════════════════════════════════════
       RANKING ROW
    ══════════════════════════════════════════════ */
    .rank-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        border-bottom: 1px solid #F3F4F6;
    }}
    .rank-number {{
        font-size: 11px;
        font-weight: 700;
        color: {TEXT_MUTED};
        width: 16px;
        flex-shrink: 0;
    }}
    .rank-label {{
        font-size: 10.5px;
        font-weight: 600;
        color: #0F172A;
        flex: 1;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .rank-value {{
        font-size: 10.5px;
        font-weight: 700;
        flex-shrink: 0;
    }}

    </style>
    """, unsafe_allow_html=True)


def render_sidebar_brand():
    st.markdown(f"""
    <div style="padding:12px 10px 8px 10px;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:4px;">
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:30px;height:30px;background:rgba(255,255,255,0.12);border-radius:8px;
                        border:1px solid rgba(255,255,255,0.18);display:flex;align-items:center;
                        justify-content:center;flex-shrink:0;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                     stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <polyline points="9 12 11 14 15 10"/>
                </svg>
            </div>
            <div>
                <div style="color:white;font-weight:700;font-size:0.88rem;line-height:1.2;">BankSurvey</div>
                <div style="color:rgba(255,255,255,0.45);font-size:0.58rem;">Customer Satisfaction 2024</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_nav_section(label):
    st.sidebar.markdown(f"""
    <div style="color:rgba(255,255,255,0.3);font-size:0.56rem;font-weight:700;
                text-transform:uppercase;letter-spacing:0.14em;
                padding:0 12px;margin:8px 0 2px 0;">{label}</div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        render_sidebar_brand()
        render_nav_section("OVERVIEW")
        st.page_link("dashboard.py", label="Executive Dashboard")
        render_nav_section("INTELLIGENCE")
        st.page_link("pages/1_Branch_Intelligence.py",     label="Branch Intelligence")
        st.page_link("pages/2_Touchpoint_Intelligence.py", label="Touchpoint Intelligence")
        st.page_link("pages/3_Customer_Intelligence.py",   label="Customer Intelligence")
        st.page_link("pages/4_Competitor_Intelligence.py", label="Competitor Intelligence")
        render_nav_section("ACTION")
        st.page_link("pages/5_Executive_Action_Center.py", label="Executive Action Center")
        render_nav_section("SUPPORT")
        st.page_link("pages/6_Data_Quality_Center.py",     label="Data Quality Center")
        st.page_link("pages/7_Report_Center.py",           label="Report Center")
        sidebar_data_coverage()
        st.sidebar.markdown(f"""
        <div style="margin:8px 6px 0;padding:6px 10px;border-top:1px solid rgba(255,255,255,0.07);
                    color:rgba(255,255,255,0.35);font-size:0.6rem;display:flex;align-items:center;gap:4px;">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2"><circle cx="12" cy="12" r="10"/>
                 <polyline points="12 6 12 12 16 14"/></svg>
            Last Updated: Jun 2024
        </div>
        """, unsafe_allow_html=True)


def render_global_header(title, subtitle, last_update="Jun 2024"):
    st.markdown(f"""
    <div class="g-header">
        <div>
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        <div class="g-header-right">
            <div class="h-ts">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
                     stroke="{TEXT_MUTED}" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12 6 12 12 16 14"/>
                </svg>
                Updated: {last_update}
            </div>
            <div class="h-btn">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                     stroke="{TEXT_SECONDARY}" stroke-width="2">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
            </div>
            <div class="h-btn">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                     stroke="{TEXT_SECONDARY}" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                </svg>
                Admin
            </div>
            <div class="h-btn primary">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
                     stroke="white" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                Export
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title, icon_svg=None, icon_color=None, icon_bg=None):
    icon_html = ""
    if icon_svg and isinstance(icon_svg, str) and icon_svg.strip().startswith('<svg'):
        bg  = icon_bg    or "rgba(200,65,11,0.1)"
        clr = icon_color or PRIMARY
        svg_colored = icon_svg.replace('stroke="currentColor"', f'stroke="{clr}"')
        icon_html = f'<div class="sec-hdr-icon" style="background:{bg};">{svg_colored}</div>'
    st.markdown(f'<div class="sec-hdr">{icon_html}{title}</div>', unsafe_allow_html=True)


def render_kpi_card(label, value, badge=None, badge_type=None,
                    icon_svg=None, icon_bg=None, icon_color=None):
    """Render a KPI card. icon_svg must be an SVG string."""
    bg  = icon_bg    or "rgba(200,65,11,0.12)"
    clr = icon_color or PRIMARY

    # Build icon HTML — only once, correctly
    if icon_svg and isinstance(icon_svg, str) and icon_svg.strip().startswith('<svg'):
        svg_colored = icon_svg.replace('stroke="currentColor"', f'stroke="{clr}"')
        icon_html = f'<div class="kpi-icon-box" style="background:{bg};">{svg_colored}</div>'
    else:
        icon_html = f'<div class="kpi-icon-box" style="background:{bg};"></div>'

    badge_html = ""
    if badge:
        bt = badge_type or "blue"
        badge_html = f'<span class="kpi-badge {bt}">{badge}</span>'

    val_s = str(value)
    vcls  = "kpi-value-sm" if len(val_s) > 6 else "kpi-value"

    st.markdown(f"""
<div class="kpi-card">
  {icon_html}
  <div class="kpi-label">{label}</div>
  <div class="{vcls}">{value}</div>
  <div>{badge_html}</div>
</div>""", unsafe_allow_html=True)


def render_alert_card(label, value, sub, alert_type="info"):
    icons = {
        "critical": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        "warning":  '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        "success":  '<polyline points="20 6 9 17 4 12"/>',
        "info":     '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    }
    colors = {"critical":"#be123c","warning":"#92400e","success":"#166534","info":"#1e40af"}
    ic_d = icons.get(alert_type, icons["info"])
    ic_c = colors.get(alert_type, "#1e40af")
    st.markdown(f"""
<div class="alert-card {alert_type}">
  <div class="alert-icon">
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="{ic_c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      {ic_d}
    </svg>
  </div>
  <div class="alert-card-body">
    <div class="alert-card-label">{label}</div>
    <div class="alert-card-value">{value}</div>
    <div class="alert-card-sub">{sub}</div>
  </div>
</div>""", unsafe_allow_html=True)


def render_ai_banner(label, main_text, sub_text=""):
    st.markdown(f"""
<div class="ai-banner">
  <div class="ai-banner-icon">AI</div>
  <div>
    <div class="ai-label">{label}</div>
    <div class="ai-text">{main_text}</div>
    {"" if not sub_text else f'<div class="ai-sub">{sub_text}</div>'}
  </div>
</div>""", unsafe_allow_html=True)


def sidebar_data_coverage(n_branches=None, total=TOTAL_BRANCHES):
    n   = n_branches or total
    pct = round(n / total * 100) if total else 100
    st.sidebar.markdown(f"""
<div style="margin:8px 6px 0;padding:9px 10px;background:rgba(255,255,255,0.07);
            border-radius:9px;border:1px solid rgba(255,255,255,0.1);">
  <div style="color:rgba(255,255,255,0.4);font-size:0.56rem;font-weight:700;
              text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Data Coverage</div>
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
    <div style="color:white;font-size:0.78rem;font-weight:700;">{n} / {total} Branches</div>
    <div style="color:{COLOR_GREEN};font-size:0.75rem;font-weight:700;">{pct}%</div>
  </div>
  <div style="background:rgba(255,255,255,0.12);border-radius:4px;height:4px;">
    <div style="background:linear-gradient(90deg,{COLOR_GREEN},{COLOR_TEAL});
                width:{pct}%;height:100%;border-radius:4px;"></div>
  </div>
</div>
""", unsafe_allow_html=True)


def plotly_layout(fig, height=220, margin=None, show_legend=True):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", size=10, color=TEXT_DARK),
        margin=margin or dict(l=4, r=4, t=18, b=4),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=8.5),
        ) if show_legend else dict(visible=False),
    )
    fig.update_xaxes(
        showgrid=False, linecolor="#D1D5DB",
        tickfont=dict(size=9, color=TEXT_DARK), zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="#F0F0F0", linecolor="rgba(0,0,0,0)",
        tickfont=dict(size=9, color=TEXT_DARK),
    )
    return fig


def nps_color(score):
    """
    Kategori warna NPS:
      90–100 → hijau terang  (#22c55e)
      75–89  → hijau lime    (#84cc16)
      60–74  → oranye        (#f97316)
      < 60   → merah         (#ef4444)
      NaN    → abu           (#94a3b8)
    """
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "#94a3b8"
    if pd.isna(s): return "#94a3b8"
    if s >= 90:  return "#22c55e"   # hijau terang
    if s >= 75:  return "#84cc16"   # hijau lime / keruh
    if s >= 60:  return "#f97316"   # oranye
    return "#ef4444"                # merah


def get_branch_status(score):
    if score < 0:   return "Critical"
    if score < 20:  return "At Risk"
    if score < 50:  return "On Track"
    return "Excellent"


def card_open(extra_style=""):
    """Return opening HTML for a manual white card (use with card_close)."""
    return f'<div class="dash-card" style="{extra_style}">'


def card_close():
    """Return closing HTML for a manual white card."""
    return '</div>'