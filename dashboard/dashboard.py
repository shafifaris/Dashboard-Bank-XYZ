import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import *
from utils.style import *

set_page_config("Executive Dashboard")
inject_global_css()
render_sidebar()

# ── Load ───────────────────────────────────────────────────────
master   = load_master()
branch   = load_branch()
provinsi = load_provinsi()
driver   = load_driver()
nps_comp = load_nps_competitor()

# ── Header ─────────────────────────────────────────────────────
render_global_header("Executive Command Center",
                     "Bank XYZ · Customer Satisfaction Survey 2024")

# ── Filters — compact full-width single row ────────────────────
fc1, fc2, fc3 = st.columns([2.5, 2.2, 1.5])
with fc1:
    provs    = ['All Provinces'] + sorted(master['provinsi'].dropna().unique().tolist())
    prov_sel = st.selectbox("PROVINSI", provs, key="ex_prov")
with fc2:
    # Ambil panel dari master, sesuai filter provinsi jika sudah dipilih
    if prov_sel != 'All Provinces':
        panel_src = master[master['provinsi'] == prov_sel]
    else:
        panel_src = master
    panel_col_name = 'panel' if 'panel' in master.columns else None
    if panel_col_name:
        panels = ['All Panels'] + sorted(panel_src[panel_col_name].dropna().unique().tolist())
    else:
        panels = ['All Panels']
    panel_sel = st.selectbox("PANEL", panels, key="ex_panel")
with fc3:
    # Kota/Kab selalu tampil — filter berdasarkan provinsi & panel yang dipilih
    kota_src = master.copy()
    if prov_sel != 'All Provinces':
        kota_src = kota_src[kota_src['provinsi'] == prov_sel]
    if panel_sel != 'All Panels' and panel_col_name:
        kota_src = kota_src[kota_src[panel_col_name] == panel_sel]
    kota_col_name = 'kota' if 'kota' in master.columns else ('KABKOTA' if 'KABKOTA' in master.columns else None)
    if kota_col_name and len(kota_src) > 0:
        kotas = ['All Cities'] + sorted(kota_src[kota_col_name].dropna().unique().tolist())
    else:
        kotas = ['All Cities']
    kota_sel = st.selectbox("KOTA/KAB", kotas, key="ex_kota")

filtered = apply_filters(master, prov_sel, panel_sel, kota_sel)

# ── Metrics ────────────────────────────────────────────────────
total_resp  = len(filtered)
# FIX: guard nps_num column presence before calling nps_score
nps_val     = nps_score(filtered['nps_num']) if 'nps_num' in filtered.columns and total_resp > 0 else 0.0
nps_val     = nps_val or 0.0
csi_val     = round(filtered['csi_num'].mean(), 1) if 'csi_num' in filtered.columns else 0.0
loy_val     = round(filtered['loyalty_num'].mean(), 1) if 'loyalty_num' in filtered.columns else 0.0
n_cabang    = filtered['cabang'].nunique() if 'cabang' in filtered.columns else 0
risk_pct    = round((filtered['nps_num'] <= RISK_THRESHOLD).sum() / total_resp * 100, 1) if total_resp > 0 and 'nps_num' in filtered.columns else 0.0

branch['status'] = branch['nps_score'].apply(get_branch_status)
n_critical  = int((branch['status'] == 'Critical').sum())
top_driver  = driver.iloc[0]['touchpoint'] if len(driver) > 0 else "N/A"
worst_prov  = provinsi.nsmallest(1, 'nps_score')['PROV'].values[0] if len(provinsi) > 0 else "N/A"
worst_nps   = float(provinsi.nsmallest(1, 'nps_score')['nps_score'].values[0]) if len(provinsi) > 0 else 0.0
best_prov   = provinsi.nlargest(1, 'nps_score')['PROV'].values[0] if len(provinsi) > 0 else "N/A"
best_nps    = float(provinsi.nlargest(1, 'nps_score')['nps_score'].values[0]) if len(provinsi) > 0 else 0.0
top_corr    = float(driver.iloc[0]['correlation']) if len(driver) > 0 else 0.0

def badge_logic(val, target):
    if val >= target:        return "Above Target", "green"
    if val >= target * 0.9:  return "On Track",     "yellow"
    return "Below Target", "red"

nps_b,  nps_bt  = badge_logic(nps_val,  NPS_TARGET)
csi_b,  csi_bt  = badge_logic(csi_val,  CSI_TARGET)
loy_b,  loy_bt  = badge_logic(loy_val,  LOYALTY_TARGET)

# ── Layout: 80% main | 20% AI Command Center ──────────────────
main_col, ai_col = st.columns([4, 1], gap="small")

with main_col:

    # ── KPI Row — full width ───────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    with k1:
        render_kpi_card("NPS Score", f"{nps_val:.1f}",
            badge=nps_b, badge_type=nps_bt,
            icon_svg=SVG_SHIELD, icon_bg="rgba(34,197,94,0.12)", icon_color=COLOR_GREEN)
    with k2:
        render_kpi_card("Cust. Satisfaction", f"{csi_val:.1f}",
            badge=csi_b, badge_type=csi_bt,
            icon_svg=SVG_STAR, icon_bg="rgba(59,130,246,0.12)", icon_color=COLOR_BLUE)
    with k3:
        render_kpi_card("Loyalty Index", f"{loy_val:.1f}",
            badge=loy_b, badge_type=loy_bt,
            icon_svg=SVG_HEART, icon_bg="rgba(139,92,246,0.12)", icon_color=COLOR_PURPLE)
    with k4:
        render_kpi_card("Customer Risk", f"{risk_pct:.1f}%",
            badge="Normal" if risk_pct < RISK_THRESHOLD else "High Risk",
            badge_type="green" if risk_pct < RISK_THRESHOLD else "red",
            icon_svg=SVG_RISK, icon_bg="rgba(239,68,68,0.12)", icon_color=COLOR_RED)
    with k5:
        render_kpi_card("Total Responses", f"{total_resp:,}",
            badge=f"{n_cabang} Branches", badge_type="blue",
            icon_svg=SVG_USERS, icon_bg="rgba(20,184,166,0.12)", icon_color=COLOR_TEAL)

    # ── Spacer / pemisah antara KPI dan Alert Row ──────────────
    st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True)

    # ── Alert Row ─────────────────────────────────────────────
    a1, a2, a3, a4 = st.columns(4, gap="small")
    with a1: render_alert_card("Critical Branches", str(n_critical), "Memerlukan perhatian segera", "critical")
    with a2: render_alert_card("Lowest NPS Province", worst_prov, f"NPS {worst_nps:.1f} — High Risk", "warning")
    with a3: render_alert_card("Top Driver", top_driver, f"r = {top_corr:.2f} — Strongest Impact", "info")
    with a4: render_alert_card("Best Province", best_prov, f"NPS {best_nps:.1f} — Excellent", "success")

    # ── Map + Rankings + Drivers (50% / 30% / 20   %) ────────────
    map_col, rank_col, drv_col = st.columns([2.5, 1.5, 1.2], gap="small")

    with map_col:
        with st.container(border=True):
            render_section_header("Indonesia Performance Map", SVG_MAP, COLOR_BLUE, "rgba(59,130,246,0.1)")
            geojson = load_geojson()
            if geojson:
                prov_data = provinsi.copy()
                prov_data = prepare_prov_for_map(prov_data, prov_col='PROV')
                if prov_sel != 'All Provinces':
                    prov_data = prov_data[prov_data['PROV'] == prov_sel]
                fig_map = px.choropleth(
                    prov_data, geojson=geojson,
                    locations='GEOJSON_NAME', featureidkey='properties.NAME_1',
                    color='nps_score',
                    color_continuous_scale=["#ef4444","#f97316","#f59e0b","#22c55e"],
                    range_color=[-20, 100],
                    hover_name='PROV',
                    hover_data={'nps_score':':.1f','n_responden':True,'GEOJSON_NAME':False},
                    labels={'nps_score':'NPS','n_responden':'Responden'},
                )
                fig_map.update_geos(
                    fitbounds="locations", visible=True,
                    showcoastlines=True, coastlinecolor="#CBD5E1",
                    showland=True,       landcolor="#F8FAFC",
                    showocean=True,      oceancolor="#EFF6FF",
                    showlakes=False, showrivers=False, showframe=False,
                    bgcolor="rgba(0,0,0,0)",
                )
                fig_map.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0,r=0,t=0,b=0), height=290,
                    coloraxis_colorbar=dict(
                        title="NPS", tickfont=dict(size=8,color=TEXT_DARK),
                        len=0.6, thickness=9, x=1.0),
                    dragmode="zoom",
                )
                st.plotly_chart(fig_map, use_container_width=True,
                    config={"displayModeBar":True,
                            "modeBarButtonsToRemove":["select2d","lasso2d"],
                            "scrollZoom":True})
                st.markdown("""
                <div style="display:flex;gap:12px;font-size:9.5px;color:#4B5563;
                            justify-content:center;margin-top:2px">
                  <span><span style="color:#22c55e">&#9632;</span> High NPS</span>
                  <span><span style="color:#f59e0b">&#9632;</span> Medium NPS</span>
                  <span><span style="color:#ef4444">&#9632;</span> Low NPS</span>
                </div>""", unsafe_allow_html=True)
            else:
                pf = provinsi.sort_values('nps_score', ascending=True).tail(15)
                fig_fb = go.Figure(go.Bar(
                    x=pf['nps_score'], y=pf['PROV'], orientation='h',
                    marker_color=[nps_color(v) for v in pf['nps_score']],
                    text=pf['nps_score'].round(1), textposition='outside',
                    textfont=dict(size=9, color=TEXT_DARK),
                ))
                fig_fb = plotly_layout(fig_fb, height=290, margin=dict(l=4,r=40,t=18,b=4))
                fig_fb.update_xaxes(title="NPS Score")
                st.plotly_chart(fig_fb, use_container_width=True, config={"displayModeBar":False})

    with rank_col:
        with st.container(border=True):
            render_section_header("NPS by Province (Top 10)", SVG_ANALYTICS, COLOR_PURPLE, "rgba(139,92,246,0.1)")
            prov_src = provinsi.copy()
            if prov_sel != 'All Provinces':
                prov_src = prov_src[prov_src['PROV'] == prov_sel]
            prov_top = prov_src.nlargest(10, 'nps_score')[['PROV','nps_score']].reset_index(drop=True)
            rows = ""
            for i, row in prov_top.iterrows():
                clr   = nps_color(row['nps_score'])
                bar_w = max(5, min(100, int((row['nps_score'] + 20) / 1.2)))
                rows += f"""<tr>
                  <td style="color:{TEXT_MUTED};width:14px;font-size:10px">{i+1}</td>
                  <td style="font-weight:600;font-size:10px;max-width:80px;
                     overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{row['PROV'][:16]}</td>
                  <td>
                    <div style="display:flex;align-items:center;gap:4px">
                      <div style="background:#F3F4F6;border-radius:3px;height:5px;width:40px;flex-shrink:0">
                        <div style="background:{clr};width:{bar_w}%;height:100%;border-radius:3px"></div>
                      </div>
                      <span style="color:{clr};font-weight:700;font-size:10px">{row['nps_score']:.1f}</span>
                    </div>
                  </td>
                </tr>"""
            st.markdown(f"""
            <div style="overflow-y:auto;max-height:290px;">
              <table class="styled-table">
                <thead><tr><th>#</th><th>Provinsi</th><th>NPS</th></tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

    with drv_col:
        with st.container(border=True):
            render_section_header("CX Drivers", SVG_TARGET, COLOR_ORANGE, "rgba(249,115,22,0.1)")
            drv_show = driver.head(8).copy()
            rows_drv = ""
            for _, row in drv_show.iterrows():
                lbl = "High" if row['abs_corr'] >= 0.3 else ("Med" if row['abs_corr'] >= 0.15 else "Low")
                clr = COLOR_RED if lbl == "High" else (COLOR_YELLOW if lbl == "Med" else COLOR_GREEN)
                bc  = "red"   if lbl == "High" else ("yellow" if lbl == "Med" else "green")
                bw  = min(100, int(row['abs_corr'] * 200))
                rows_drv += f"""<tr>
                  <td style="font-weight:600;font-size:9.5px;max-width:65px;
                     overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{row['touchpoint'][:14]}</td>
                  <td><div style="background:#F3F4F6;border-radius:3px;height:4px;width:30px">
                    <div style="background:{clr};width:{bw}%;height:100%;border-radius:3px"></div>
                  </div></td>
                  <td style="font-size:9.5px;font-weight:600;color:{TEXT_DARK}">{row['correlation']:.2f}</td>
                  <td><span class="badge badge-{bc}">{lbl}</span></td>
                </tr>"""
            st.markdown(f"""
            <div style="overflow-y:auto;max-height:290px;">
              <table class="styled-table">
                <thead><tr><th>Driver</th><th></th><th>r</th><th>Lvl</th></tr></thead>
                <tbody>{rows_drv}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

    # ── Bottom Row: Risk / Opportunity / Benchmark / Data Quality
    b1, b2, b3, b4 = st.columns(4, gap="small")

    with b1:
        with st.container(border=True):
            render_section_header("Top Risk Provinces", SVG_WARNING, COLOR_RED, "rgba(239,68,68,0.1)")
            prov_risk = provinsi.copy()
            if prov_sel != 'All Provinces':
                prov_risk = prov_risk[prov_risk['PROV'] == prov_sel]
            rows = ""
            for i, row in enumerate(prov_risk.nsmallest(5,'nps_score').itertuples(), 1):
                rc  = "red" if row.nps_score < 20 else "yellow"
                clr = COLOR_RED if row.nps_score < 20 else COLOR_YELLOW
                bw  = max(5, min(100, int((row.nps_score + 20) / 1.4)))
                rows += f"""<tr>
                  <td style="color:{TEXT_MUTED};font-size:10px;width:14px">{i}</td>
                  <td style="font-weight:600;font-size:10px;max-width:75px;
                      overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{row.PROV[:13]}</td>
                  <td>
                    <div style="display:flex;align-items:center;gap:3px">
                      <div style="background:#F3F4F6;border-radius:3px;height:4px;width:30px;flex-shrink:0">
                        <div style="background:{clr};width:{bw}%;height:100%;border-radius:3px"></div>
                      </div>
                      <span style="color:{clr};font-weight:700;font-size:9.5px">{row.nps_score:.1f}</span>
                    </div>
                  </td>
                  <td><span class="badge badge-{rc}">{"Crit" if row.nps_score < 20 else "Risk"}</span></td>
                </tr>"""
            st.markdown(f"""<div style="overflow-y:auto;max-height:180px;">
              <table class="styled-table">
                <thead><tr><th>#</th><th>Provinsi</th><th>NPS</th><th>Status</th></tr></thead>
                <tbody>{rows}</tbody>
              </table></div>""", unsafe_allow_html=True)

    with b2:
        with st.container(border=True):
            render_section_header("Opportunity Provinces", SVG_CHECK, COLOR_GREEN, "rgba(34,197,94,0.1)")
            prov_opp = provinsi.copy()
            if prov_sel != 'All Provinces':
                prov_opp = prov_opp[prov_opp['PROV'] == prov_sel]
            rows = ""
            for i, row in enumerate(prov_opp.nlargest(5,'nps_score').itertuples(), 1):
                oc  = "green" if row.nps_score >= NPS_TARGET else "yellow"
                clr = COLOR_GREEN if row.nps_score >= NPS_TARGET else COLOR_YELLOW
                bw  = max(5, min(100, int((row.nps_score + 20) / 1.4)))
                rows += f"""<tr>
                  <td style="color:{TEXT_MUTED};font-size:10px;width:14px">{i}</td>
                  <td style="font-weight:600;font-size:10px;max-width:75px;
                      overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{row.PROV[:13]}</td>
                  <td>
                    <div style="display:flex;align-items:center;gap:3px">
                      <div style="background:#F3F4F6;border-radius:3px;height:4px;width:30px;flex-shrink:0">
                        <div style="background:{clr};width:{bw}%;height:100%;border-radius:3px"></div>
                      </div>
                      <span style="color:{clr};font-weight:700;font-size:9.5px">{row.nps_score:.1f}</span>
                    </div>
                  </td>
                  <td><span class="badge badge-{oc}">{"Strong" if row.nps_score >= NPS_TARGET else "Grow"}</span></td>
                </tr>"""
            st.markdown(f"""<div style="overflow-y:auto;max-height:180px;">
              <table class="styled-table">
                <thead><tr><th>#</th><th>Provinsi</th><th>NPS</th><th>Opp</th></tr></thead>
                <tbody>{rows}</tbody>
              </table></div>""", unsafe_allow_html=True)

    with b3:
        with st.container(border=True):
            render_section_header("Benchmark Comparison", SVG_ANALYTICS, COLOR_BLUE, "rgba(59,130,246,0.1)")
            comp_mean = float(nps_comp[nps_comp['bank'] != 'Bank XYZ']['nps_score'].mean()) if len(nps_comp) > 1 else 0
            metrics_bench = [
                ("NPS",     nps_val,  comp_mean, NPS_TARGET),
                ("CSI",     csi_val,  76.5,      CSI_TARGET),
                ("Loyalty", loy_val,  70.8,      LOYALTY_TARGET),
            ]
            rows = ""
            for m, xyz, comp, tgt in metrics_bench:
                diff    = xyz - comp
                dc      = COLOR_GREEN if diff >= 0 else COLOR_RED
                ds      = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
                pct_bar = min(100, max(0, int(xyz / max(tgt * 1.2, 1) * 100)))
                rows += f"""<tr>
                  <td style="font-weight:600;font-size:10px">{m}</td>
                  <td>
                    <div style="background:#F3F4F6;border-radius:3px;height:5px;width:48px;margin-bottom:2px">
                      <div style="background:{COLOR_BLUE};width:{pct_bar}%;height:100%;border-radius:3px"></div>
                    </div>
                    <span style="font-size:9.5px;font-weight:700;color:{TEXT_DARK}">{xyz:.1f}</span>
                  </td>
                  <td style="color:{TEXT_MUTED};font-size:9.5px">{comp:.1f}</td>
                  <td style="color:{dc};font-weight:700;font-size:9.5px">{ds}</td>
                </tr>"""
            st.markdown(f"""<table class="styled-table" style="margin-bottom:4px;">
              <thead><tr><th>Metric</th><th>XYZ</th><th>Comp</th><th>vs Avg</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

    with b4:
        with st.container(border=True):
            render_section_header("Data Quality", SVG_VERIFIED, COLOR_TEAL, "rgba(20,184,166,0.1)")
            completeness = round(filtered['nps_num'].notna().sum() / total_resp * 100, 1) if total_resp > 0 else 0.0
            n_promoters  = int((filtered['nps_num'] >= 9).sum())
            n_passives   = int(((filtered['nps_num'] == 7) | (filtered['nps_num'] == 8)).sum())
            n_detractors = int((filtered['nps_num'] <= 6).sum())

            # Progress bars for Coverage, Completeness, Confidence
            dq_items = [
                ("Coverage",     round(n_cabang/TOTAL_BRANCHES*100,1), COLOR_TEAL),
                ("Completeness", completeness,                          COLOR_GREEN if completeness>=90 else COLOR_YELLOW),
                ("Confidence",   95.0,                                  COLOR_GREEN),
            ]
            for lbl, val, clr in dq_items:
                st.markdown(f"""
                <div style="margin-bottom:6px">
                  <div style="display:flex;justify-content:space-between;margin-bottom:2px">
                    <span style="font-size:9.5px;color:{TEXT_MUTED};font-weight:600">{lbl}</span>
                    <span style="font-size:9.5px;font-weight:700;color:{clr}">{val:.1f}%</span>
                  </div>
                  <div style="background:#F3F4F6;border-radius:4px;height:5px">
                    <div style="background:{clr};width:{min(100,val)}%;height:100%;border-radius:4px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div style="height:4px"></div>', unsafe_allow_html=True)

            rows_q = [
                ("Total Responses",  f"{total_resp:,}",               TEXT_DARK),
                ("Promoters",        f"{n_promoters:,}",               COLOR_GREEN),
                ("Passives",         f"{n_passives:,}",                COLOR_YELLOW),
                ("Detractors",       f"{n_detractors:,}",              COLOR_RED),
            ]
            for k, v, c in rows_q:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:2px 0;
                    border-bottom:1px solid #F3F4F6;font-size:10px;">
                    <span style="color:{TEXT_MUTED}">{k}</span>
                    <span style="font-weight:700;color:{c}">{v}</span>
                </div>""", unsafe_allow_html=True)

# ── AI Command Center — permanent right panel ──────────────────
with ai_col:
    # ── AI Executive Summary ───────────────────────────────────
    with st.container(border=True):
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{PRIMARY},{PRIMARY_LIGHT});
                    border-radius:10px;padding:10px 12px;margin-bottom:10px;">
          <div style="display:flex;align-items:center;gap:7px;margin-bottom:2px">
            <div style="width:22px;height:22px;background:rgba(255,255,255,0.2);
                        border-radius:6px;display:flex;align-items:center;justify-content:center;">
              {SVG_BOLT.replace('stroke="currentColor"','stroke="white"')}
            </div>
            <div style="font-weight:700;font-size:12px;color:white;">Executive AI Copilot</div>
          </div>
          <div style="font-size:9px;color:rgba(255,255,255,0.75);">Strategic insights &amp; recommendations</div>
        </div>

        <div style="font-size:9px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.09em;color:{TEXT_MUTED};margin-bottom:6px;">INSIGHT OF THE DAY</div>
        <div style="background:#FFF7ED;border:1px solid #FED7AA;border-left:3px solid {PRIMARY};
                    border-radius:8px;padding:8px 10px;margin-bottom:10px;">
          <div style="font-size:10.5px;font-weight:600;color:#0F172A;line-height:1.4;">
            {top_driver} menjadi driver utama penurunan NPS. {worst_prov} memerlukan perhatian segera.
          </div>
        </div>

        <div style="font-size:9px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.09em;color:{TEXT_MUTED};margin-bottom:6px;">KEY FINDINGS</div>
        """, unsafe_allow_html=True)

        findings = [
            (COLOR_RED,    f"{worst_prov} — risiko tertinggi",    f"NPS {worst_nps:.1f}"),
            (COLOR_YELLOW, f"{top_driver} — driver utama",         f"r = {top_corr:.2f}"),
            (COLOR_GREEN,  f"{best_prov} — performa terbaik",      f"NPS {best_nps:.1f}"),
            (COLOR_BLUE,   f"{n_critical} cabang kritis",          "Perlu tindakan segera"),
        ]
        for clr, t, s in findings:
            st.markdown(f"""
            <div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid #F3F4F6;align-items:flex-start;">
              <div style="width:7px;height:7px;background:{clr};border-radius:50%;
                          margin-top:3px;flex-shrink:0;"></div>
              <div>
                <div style="font-size:10.5px;font-weight:600;color:#0F172A;">{t}</div>
                <div style="font-size:9px;color:{TEXT_MUTED};">{s}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:9px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.09em;color:{TEXT_MUTED};margin:10px 0 6px;">TOP OPPORTUNITIES</div>
        """, unsafe_allow_html=True)

        for _, row in driver.head(3).iterrows():
            est = round(row['abs_corr'] * 15, 1)
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:5px 0;border-bottom:1px solid #F3F4F6;">
              <div style="display:flex;align-items:center;gap:6px;">
                <div style="width:18px;height:18px;background:rgba(34,197,94,0.12);
                            border-radius:5px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                  {SVG_TREND_UP.replace('stroke="currentColor"',f'stroke="{COLOR_GREEN}"')}
                </div>
                <div>
                  <div style="font-size:10.5px;font-weight:600;color:#0F172A;">{row['touchpoint']}</div>
                  <div style="font-size:9px;color:{TEXT_MUTED};">Est. NPS increase</div>
                </div>
              </div>
              <span style="font-size:11px;color:{COLOR_GREEN};font-weight:700;flex-shrink:0;">+{est}</span>
            </div>""", unsafe_allow_html=True)

    # ── AI Assistant ───────────────────────────────────────────
    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
          <div style="width:22px;height:22px;background:linear-gradient(135deg,{PRIMARY},{PRIMARY_LIGHT});
                      border-radius:6px;display:flex;align-items:center;justify-content:center;">
            {SVG_BOLT.replace('stroke="currentColor"','stroke="white"')}
          </div>
          <div>
            <div style="font-weight:700;font-size:11.5px;color:#0F172A;">Ask AI</div>
            <div style="font-size:8.5px;color:{TEXT_MUTED};">Tanya tentang survei 2024</div>
          </div>
        </div>
        <div style="font-size:9px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.08em;color:{TEXT_MUTED};margin-bottom:6px;">QUICK QUESTIONS</div>
        """, unsafe_allow_html=True)

        if 'exec_chat' not in st.session_state:
            st.session_state.exec_chat = [
                {"role": "ai", "content": "Halo! Tanya apa saja tentang performa survei 2024."}]

        quick_qs = ["Top 3 risiko?", "Provinsi terbaik?", "Driver NPS?"]
        for qi, q in enumerate(quick_qs):
            if st.button(q, key=f"ex_q{qi}", use_container_width=True):
                ans = {
                    "Top 3 risiko?":
                        f"Risiko: {', '.join(provinsi.nsmallest(3,'nps_score')['PROV'].tolist())}.",
                    "Provinsi terbaik?":
                        f"{best_prov} — NPS {best_nps:.1f}.",
                    "Driver NPS?":
                        f"Driver utama: {top_driver} (r={top_corr:.2f}).",
                }
                st.session_state.exec_chat += [
                    {"role":"user","content":q},
                    {"role":"ai",  "content":ans.get(q,"")}]
                st.rerun()

        if len(st.session_state.exec_chat) > 1:
            chat_html = '<div class="chat-box">'
            for msg in st.session_state.exec_chat[-6:]:
                chat_html += f'<div class="chat-{"user" if msg["role"]=="user" else "ai"}">{msg["content"]}</div>'
            st.markdown(chat_html + '</div>', unsafe_allow_html=True)

        user_q = st.text_input("Ketik pertanyaan...", key="ex_uq",
                               label_visibility="collapsed",
                               placeholder="Ketik pertanyaan...")
        if user_q:
            st.session_state.exec_chat += [
                {"role":"user","content":user_q},
                {"role":"ai",  "content":f"NPS {nps_val:.1f}, {n_critical} cabang kritis. Top driver: {top_driver}."}]
            st.rerun()