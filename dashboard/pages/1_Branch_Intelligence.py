import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import *
from utils.style import *

set_page_config("Branch Intelligence", "🏢")
inject_global_css()
render_sidebar()

# ── Load Data ──────────────────────────────────────────────────
branch   = load_branch()
provinsi = load_provinsi()
master   = load_master()
driver   = load_driver()

branch['status'] = branch['nps_score'].apply(get_branch_status)

n_total    = len(branch)
n_healthy  = int(branch['status'].isin(['Excellent', 'On Track']).sum())
n_warning  = int((branch['status'] == 'At Risk').sum())
n_critical = int((branch['status'] == 'Critical').sum())
avg_nps    = round(branch['nps_score'].mean(), 1)

# Pre-compute shared variables
crit_list     = branch[branch['status'] == 'Critical'].nsmallest(5, 'nps_score') if n_critical > 0 else branch.nsmallest(2, 'nps_score')
top_drv_name  = driver.iloc[0]['touchpoint'] if len(driver) > 0 else "Customer Service"
top_drv2_name = driver.iloc[1]['touchpoint'] if len(driver) > 1 else "Waiting Time"
top_drv_est   = round(driver.iloc[0]['abs_corr'] * 15, 1) if len(driver) > 0 else 5.0
top_drv2_est  = round(driver.iloc[1]['abs_corr'] * 15, 1) if len(driver) > 1 else 3.0
worst_prov    = provinsi.nsmallest(1, 'nps_score')['PROV'].values[0] if len(provinsi) > 0 else "N/A"
crit_name     = crit_list.iloc[0]['CABANG'] if len(crit_list) > 0 else "N/A"
crit_prov     = crit_list.iloc[0]['PROV']   if len(crit_list) > 0 else "N/A"

main_col, ai_col = st.columns([3, 1], gap="small")

with main_col:
    st.markdown('<div class="page-title">Branch Intelligence Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Monitor branch performance, identify risks, and take action</div>', unsafe_allow_html=True)

    # ── Alert Banner ───────────────────────────────────────────
    render_ai_banner(
        "Branch Summary",
        f"{n_critical} branch memerlukan perhatian segera.",
        f"{crit_name} ({crit_prov}) mencatat NPS terendah. Driver utama: {top_drv_name}."
    )

    # ── Filters ────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4, gap="small")
    with f1:
        provs    = ['All Provinces'] + sorted(branch['PROV'].dropna().unique().tolist())
        prov_sel = st.selectbox("PROVINSI", provs, key="br_prov")
    with f2:
        kotas_all = branch['KABKOTA'].dropna().unique().tolist() if 'KABKOTA' in branch.columns else []
        kotas     = ['All Cities'] + sorted(
            branch[branch['PROV'] == prov_sel]['KABKOTA'].dropna().unique().tolist()
            if prov_sel != 'All Provinces' else kotas_all)
        kota_sel  = st.selectbox("KOTA/KAB", kotas, key="br_kota")
    with f3:
        status_sel = st.selectbox("STATUS", ['All Status', 'Excellent', 'On Track', 'At Risk', 'Critical'], key="br_status")
    with f4:
        sort_sel   = st.selectbox("SORT BY", ['NPS Score', 'CSI', 'Loyalty', 'Responden'], key="br_sort")

    br = branch.copy()
    if prov_sel != 'All Provinces':                            br = br[br['PROV'] == prov_sel]
    if kota_sel != 'All Cities' and 'KABKOTA' in br.columns:  br = br[br['KABKOTA'] == kota_sel]
    if status_sel != 'All Status':                             br = br[br['status'] == status_sel]
    sort_map = {'NPS Score': 'nps_score', 'CSI': 'csi_num', 'Loyalty': 'loyalty_num', 'Responden': 'n_responden'}
    br = br.sort_values(sort_map[sort_sel], ascending=False)

    # ── KPI Cards ──────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    with k1: render_kpi_card("Total Branches",     str(n_total),    badge="+2 vs Apr 2026",                       badge_type="blue",   icon_svg=SVG_BRANCH,   icon_bg="rgba(200,65,11,0.1)",  icon_color=PRIMARY)
    with k2: render_kpi_card("Healthy Branches",   str(n_healthy),  badge=f"{round(n_healthy/n_total*100,1)}%",   badge_type="green",  icon_svg=SVG_SHIELD,   icon_bg="rgba(34,197,94,0.1)",  icon_color=COLOR_GREEN)
    with k3: render_kpi_card("At Risk Branches",   str(n_warning),  badge=f"{round(n_warning/n_total*100,1)}%",   badge_type="yellow", icon_svg=SVG_WARNING,  icon_bg="rgba(245,158,11,0.1)", icon_color=COLOR_YELLOW)
    with k4: render_kpi_card("Critical Branches",  str(n_critical), badge=f"{round(n_critical/n_total*100,1)}%",  badge_type="red",    icon_svg=SVG_RISK,     icon_bg="rgba(239,68,68,0.1)",  icon_color=COLOR_RED)
    with k5: render_kpi_card("Avg NPS All Branch", f"{avg_nps:.1f}", badge="vs Apr 2026 +2.3",                    badge_type="blue",   icon_svg=SVG_TREND_UP, icon_bg="rgba(59,130,246,0.1)", icon_color=COLOR_BLUE)

    # ── Row 2: Map + Regional + Distribution ───────────────────
    map_col2, reg_col, dist_col = st.columns([2.5, 1.5, 1.2], gap="small")

    with map_col2:
        with st.container(border=True):
            render_section_header("Branch Risk Map", SVG_MAP, COLOR_BLUE, "rgba(59,130,246,0.1)")
            geojson = load_geojson()
            prov_branch_stats = branch.groupby('PROV').agg(
                nps_score  =('nps_score', 'mean'),
                n_branch   =('CABANG', 'count'),
                n_critical =('status', lambda x: (x == 'Critical').sum())
            ).reset_index()
            if geojson:
                prov_map_data = prepare_prov_for_map(prov_branch_stats, prov_col='PROV')
                if prov_sel != 'All Provinces':
                    prov_map_data = prov_map_data[prov_map_data['PROV'] == prov_sel]
                fig_map = px.choropleth(
                    prov_map_data, geojson=geojson, locations='GEOJSON_NAME',
                    featureidkey='properties.NAME_1', color='nps_score',
                    color_continuous_scale=["#ef4444", "#f97316", "#f59e0b", "#22c55e"],
                    range_color=[-20, 100], hover_name='PROV',
                    hover_data={'nps_score': ':.1f', 'n_branch': True, 'n_critical': True, 'GEOJSON_NAME': False},
                    labels={'nps_score': 'NPS', 'n_branch': 'Cabang', 'n_critical': 'Critical'},
                )
                fig_map.update_geos(
                fitbounds="locations", visible=True,
                showcoastlines=True, coastlinecolor="#E5E7EB",
                showland=True,  landcolor="#F9FAFB",
                showocean=True, oceancolor="#EFF6FF",
                showlakes=False, showrivers=False, showframe=False,
                bgcolor="rgba(0,0,0,0)",
                projection_type="mercator",
                lataxis_range=[-11, 6],
                lonaxis_range=[94, 142],
                )
                fig_map.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=0, b=0), height=300,
                    coloraxis_colorbar=dict(
                        title="NPS", tickfont=dict(size=8, color=TEXT_DARK),
                        len=0.6, thickness=8, x=1.0),
                    dragmode="zoom")
                st.plotly_chart(fig_map, use_container_width=True,
                    config={"displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            "scrollZoom": True})
                st.markdown("""
                <div style="display:flex;gap:14px;font-size:0.68rem;color:#4B5563;
                            justify-content:center;margin-top:0px;padding-bottom:2px">
                  <span><span style="color:#22c55e">&#9632;</span>&nbsp;Low Risk (NPS &ge; 70)</span>
                  <span><span style="color:#f59e0b">&#9632;</span>&nbsp;Medium Risk (NPS 30–70)</span>
                  <span><span style="color:#ef4444">&#9632;</span>&nbsp;High Risk (NPS &lt; 30)</span>
                </div>""", unsafe_allow_html=True)
            else:
                top_prov_fb = prov_branch_stats.sort_values('nps_score', ascending=True).tail(15)
                fig_fb = go.Figure(go.Bar(
                    x=top_prov_fb['nps_score'], y=top_prov_fb['PROV'], orientation='h',
                    marker_color=[nps_color(v) for v in top_prov_fb['nps_score']],
                    text=top_prov_fb['nps_score'].round(1), textposition='outside',
                    textfont=dict(size=9, color=TEXT_DARK),
                ))
                fig_fb = plotly_layout(fig_fb, height=300, margin=dict(l=4, r=40, t=4, b=4))
                fig_fb.update_xaxes(title="NPS Score")
                st.plotly_chart(fig_fb, use_container_width=True, config={"displayModeBar": False})

    with reg_col:
        with st.container(border=True):
            render_section_header("Regional Performance", SVG_ANALYTICS, COLOR_PURPLE, "rgba(139,92,246,0.1)")
            top5 = provinsi.nlargest(5,  'nps_score')[['PROV', 'nps_score']]
            bot5 = provinsi.nsmallest(5, 'nps_score')[['PROV', 'nps_score']]
            st.markdown(f"<div style='font-size:0.7rem;font-weight:600;color:{COLOR_GREEN};margin-bottom:2px'>Top 5</div>", unsafe_allow_html=True)
            rows_top = "".join([
                f"<tr><td style='color:{TEXT_MUTED}'>{i}</td>"
                f"<td style='font-size:0.76rem;color:{TEXT_DARK}'>{r.PROV}</td>"
                f"<td style='color:{nps_color(r.nps_score)};font-weight:700'>{r.nps_score:.1f}</td></tr>"
                for i, r in enumerate(top5.itertuples(), 1)])
            st.markdown(f"<table class='styled-table'><thead><tr><th>#</th><th>Province</th><th>NPS</th></tr></thead><tbody>{rows_top}</tbody></table>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.7rem;font-weight:600;color:{COLOR_RED};margin:4px 0 2px'>Bottom 5</div>", unsafe_allow_html=True)
            rows_bot = "".join([
                f"<tr><td style='color:{TEXT_MUTED}'>{i}</td>"
                f"<td style='font-size:0.76rem;color:{TEXT_DARK}'>{r.PROV}</td>"
                f"<td style='color:{nps_color(r.nps_score)};font-weight:700'>{r.nps_score:.1f}</td></tr>"
                for i, r in enumerate(bot5.itertuples(), 1)])
            st.markdown(f"<table class='styled-table'><thead><tr><th>#</th><th>Province</th><th>NPS</th></tr></thead><tbody>{rows_bot}</tbody></table>", unsafe_allow_html=True)

    with dist_col:
        with st.container(border=True):
            render_section_header("Branch Distribution", SVG_USERS, COLOR_TEAL, "rgba(20,184,166,0.1)")
            for lbl, n_val, clr, bg_clr in [
                ("Healthy",  n_healthy,  COLOR_GREEN,  "rgba(34,197,94,0.08)"),
                ("At Risk",  n_warning,  COLOR_YELLOW, "rgba(245,158,11,0.08)"),
                ("Critical", n_critical, COLOR_RED,    "rgba(239,68,68,0.08)"),
            ]:
                pct   = round(n_val / n_total * 100, 1) if n_total > 0 else 0
                bar_w = max(pct, 2)
                st.markdown(f"""
                <div style="background:{bg_clr};border-radius:5px;padding:6px 8px;margin-bottom:5px">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
                    <div style="display:flex;align-items:center;gap:5px">
                      <div style="width:6px;height:6px;background:{clr};border-radius:50%"></div>
                      <span style="font-size:0.75rem;font-weight:600;color:{TEXT_DARK}">{lbl}</span>
                    </div>
                    <span style="font-size:0.75rem;font-weight:700;color:{TEXT_DARK}">{n_val} <span style="color:{TEXT_MUTED};font-weight:400">({pct}%)</span></span>
                  </div>
                  <div style="background:rgba(0,0,0,0.06);border-radius:3px;height:3px">
                    <div style="background:{clr};width:{bar_w}%;height:100%;border-radius:3px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align:center;margin-top:4px;padding:6px 0">
              <span style="font-size:1.3rem;font-weight:700;color:{TEXT_DARK}">{n_total}</span>
              <span style="font-size:0.68rem;color:{TEXT_MUTED};display:block">Total Branches</span>
            </div>""", unsafe_allow_html=True)

    # ── Row 3: Critical (kiri, digabung AI Rec) + Opportunity (kanan) ──
    crit_col, opp_col = st.columns(2, gap="small")

    with crit_col:
        # Critical Branches + AI Recommendation digabung dalam satu kolom
        with st.container(border=True):
            st.markdown(f"<div style='border-left:3px solid {COLOR_RED};padding-left:8px;margin-bottom:6px'>", unsafe_allow_html=True)
            render_section_header("Critical Branches", SVG_WARNING, COLOR_RED, "rgba(239,68,68,0.1)")
            st.markdown("</div>", unsafe_allow_html=True)
            crit_br = branch[branch['status'] == 'Critical'].nsmallest(5, 'nps_score')
            if len(crit_br) == 0:
                crit_br = branch.nsmallest(5, 'nps_score')
            rows_crit = "".join([
                f"<tr style='background:{'#fff5f5' if r.nps_score < 0 else 'transparent'}'>"
                f"<td style='color:{TEXT_MUTED}'>{i}</td>"
                f"<td><div style='font-weight:600;font-size:0.78rem;color:{TEXT_DARK}'>{r.CABANG}</div>"
                f"<div style='font-size:0.68rem;color:{TEXT_MUTED}'>{r.PROV}</div></td>"
                f"<td style='color:{COLOR_RED};font-weight:700'>{r.nps_score:.1f}</td>"
                f"<td style='font-weight:600;color:{TEXT_DARK}'>{r.csi_num:.2f}</td>"
                f"<td style='font-weight:600;color:{TEXT_DARK}'>{r.loyalty_num:.2f}</td></tr>"
                for i, r in enumerate(crit_br.itertuples(), 1)])
            st.markdown(f"<table class='styled-table'><thead><tr><th>#</th><th>Branch</th><th>NPS</th><th>CSI</th><th>Loyalty</th></tr></thead><tbody>{rows_crit}</tbody></table>", unsafe_allow_html=True)

        # AI Recommendation langsung di bawah Critical, masih kolom kiri
        with st.container(border=True):
            render_section_header("AI Recommendation", SVG_BOLT, PRIMARY, "rgba(200,65,11,0.1)")
            n_show   = min(len(crit_list), 2)
            drv_names = [top_drv_name, top_drv2_name]
            drv_ests  = [top_drv_est,  top_drv2_est]
            rc_cols   = st.columns(n_show, gap="small") if n_show > 1 else [st.container()]
            for col_rc, idx in zip(rc_cols, range(n_show)):
                with col_rc:
                    row = crit_list.iloc[idx]
                    st.markdown(f"""
                    <div class="dash-card" style="border-top:3px solid {PRIMARY};">
                      <div style="font-size:0.65rem;font-weight:700;color:{PRIMARY};text-transform:uppercase;letter-spacing:0.06em">Priority {idx+1} — Immediate Action</div>
                      <div style="font-weight:700;font-size:0.84rem;margin:3px 0 1px;color:{TEXT_DARK}">{row['CABANG']}</div>
                      <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-bottom:6px">{row['PROV']}</div>
                      <div style="display:flex;justify-content:space-between;font-size:0.75rem">
                        <div><span style="color:{TEXT_MUTED}">Root Cause</span><br><b style="color:{TEXT_DARK}">{drv_names[idx]}</b></div>
                        <div><span style="color:{TEXT_MUTED}">Est. NPS</span><br><b style="color:{COLOR_GREEN}">+{drv_ests[idx]}</b></div>
                        <div><span style="color:{TEXT_MUTED}">Effort</span><br><b style="color:{COLOR_YELLOW}">Medium</b></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

    with opp_col:
        with st.container(border=True):
            render_section_header("Top Opportunity Branches", SVG_CHECK, COLOR_GREEN, "rgba(34,197,94,0.1)")
            top_br = br.nlargest(5, 'nps_score')
            rows_opp = "".join([
                f"<tr><td style='color:{TEXT_MUTED}'>{i}</td>"
                f"<td><div style='font-weight:600;font-size:0.78rem;color:{TEXT_DARK}'>{r.CABANG}</div>"
                f"<div style='font-size:0.68rem;color:{TEXT_MUTED}'>{r.PROV}</div></td>"
                f"<td style='color:{nps_color(r.nps_score)};font-weight:700'>{r.nps_score:.1f}</td>"
                f"<td style='font-weight:600;color:{TEXT_DARK}'>{r.csi_num:.2f}</td>"
                f"<td style='font-weight:600;color:{TEXT_DARK}'>{r.loyalty_num:.2f}</td></tr>"
                for i, r in enumerate(top_br.itertuples(), 1)])
            st.markdown(f"<table class='styled-table'><thead><tr><th>#</th><th>Branch</th><th>NPS</th><th>CSI</th><th>Loyalty</th></tr></thead><tbody>{rows_opp}</tbody></table>", unsafe_allow_html=True)

    # ── Row 4: Heatmap + RCA + Driver Impact ───────────────────
    heat_col, rca_col, drv2_col = st.columns([1.5, 1, 1], gap="small")

    with heat_col:
        with st.container(border=True):
            render_section_header("Branch Risk Heatmap", SVG_ANALYTICS, COLOR_ORANGE, "rgba(249,115,22,0.1)")
            st.markdown(
                f"<div style='font-size:0.65rem;color:{TEXT_MUTED};margin:-2px 0 4px'>"
                f"Menampilkan {min(8, len(branch))} cabang dengan NPS terendah dari total {n_total} cabang</div>",
                unsafe_allow_html=True)
            heat_data = branch.nsmallest(8, 'nps_score')[['CABANG', 'nps_score', 'csi_num', 'loyalty_num']].copy()
            z_data    = heat_data[['nps_score', 'csi_num', 'loyalty_num']].values
            fig_heat  = go.Figure(go.Heatmap(
                z=z_data, x=['NPS', 'CSI', 'Loyalty'], y=heat_data['CABANG'].tolist(),
                colorscale=[[0, '#ef4444'], [0.4, '#f59e0b'], [0.7, '#22c55e'], [1, '#16a34a']],
                text=[[f"{v:.1f}" for v in row] for row in z_data],
                texttemplate='%{text}',
                textfont=dict(size=9),
                showscale=False,
            ))
            fig_heat = plotly_layout(fig_heat, height=200, margin=dict(l=0, r=0, t=4, b=0))
            fig_heat.update_layout(margin=dict(l=105, r=4, t=4, b=4))
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    with rca_col:
        with st.container(border=True):
            render_section_header("Root Cause Analysis", SVG_TARGET, COLOR_RED, "rgba(239,68,68,0.1)")
            st.markdown(
                f"<div style='font-size:0.65rem;color:{TEXT_MUTED};margin:-2px 0 6px'>"
                f"% kontribusi terhadap ketidakpuasan pelanggan</div>",
                unsafe_allow_html=True)
            if len(driver) > 0:
                drv_data        = driver.head(5).copy()
                drv_data['pct'] = (drv_data['abs_corr'] / drv_data['abs_corr'].sum() * 100).round(1)
                colors_rca      = [COLOR_RED, COLOR_YELLOW, COLOR_ORANGE, COLOR_GREEN, COLOR_BLUE]
                for i, row in drv_data.iterrows():
                    color = colors_rca[min(i, 4)]
                    st.markdown(f"""
                    <div style="margin-bottom:5px">
                      <div style="display:flex;justify-content:space-between;margin-bottom:2px">
                        <span style="font-size:0.75rem;font-weight:500;color:{TEXT_DARK}">{row['touchpoint']}</span>
                        <span style="font-size:0.75rem;font-weight:700;color:{color}">{row['pct']:.1f}%</span>
                      </div>
                      <div style="background:#F3F4F6;border-radius:3px;height:4px">
                        <div style="background:{color};width:{row['pct']}%;height:100%;border-radius:3px"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

    with drv2_col:
        with st.container(border=True):
            render_section_header("Driver Impact on NPS", SVG_BOLT, COLOR_YELLOW, "rgba(245,158,11,0.1)")
            if len(driver) > 0:
                drv5     = driver.head(5)
                rows_drv = "".join([
                    f"<tr>"
                    f"<td style='font-size:0.75rem;font-weight:500;color:{TEXT_DARK}'>{r['touchpoint']}</td>"
                    f"<td style='font-weight:700;color:{PRIMARY}'>{r['correlation']:.2f}</td>"
                    f"<td><span class='badge {'badge-red' if r['abs_corr']>=0.3 else ('badge-yellow' if r['abs_corr']>=0.15 else 'badge-green')}'>{'High' if r['abs_corr']>=0.3 else ('Medium' if r['abs_corr']>=0.15 else 'Low')}</span></td>"
                    f"</tr>"
                    for _, r in drv5.iterrows()])
                st.markdown(f"<table class='styled-table'><thead><tr><th>Driver</th><th>r</th><th>Level</th></tr></thead><tbody>{rows_drv}</tbody></table>", unsafe_allow_html=True)

# ── AI Column ──────────────────────────────────────────────────
with ai_col:
    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
          <div style="width:24px;height:24px;background:linear-gradient(135deg,{PRIMARY},{PRIMARY_LIGHT});
                      border-radius:6px;display:flex;align-items:center;justify-content:center;">
            <span style="color:white;font-size:0.72rem;font-weight:700">AI</span>
          </div>
          <div style="font-weight:700;font-size:11px;color:{TEXT_DARK};">AI Assistant</div>
        </div>
        <div style="font-size:10px;color:{TEXT_MUTED};margin-bottom:6px">Tanya seputar performa cabang.</div>
        <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;
                    color:{TEXT_MUTED};margin-bottom:4px">SUGGESTED QUESTIONS</div>
        """, unsafe_allow_html=True)

        suggested_br = [
            "Cabang mana yang perlu perhatian segera?",
            "Apa driver utama yang mempengaruhi NPS?",
            "Provinsi mana yang performa terbaik?",
        ]

        if 'br_chat' not in st.session_state:
            st.session_state.br_chat = []

        for q in suggested_br:
            if st.button(q, key=f"br_sq_{q}", use_container_width=True):
                st.session_state.br_chat.append({"role": "user", "content": q})
                if "perhatian" in q.lower():
                    resp = f"{n_critical} cabang kritis. Teratas: {crit_list.iloc[0]['CABANG'] if len(crit_list) > 0 else 'N/A'}."
                elif "driver" in q.lower():
                    resp = f"Driver utama: {', '.join(driver.head(3)['touchpoint'].tolist())}."
                else:
                    best = provinsi.nlargest(1, 'nps_score')
                    resp = f"Terbaik: {best['PROV'].values[0]} NPS {best['nps_score'].values[0]:.1f}."
                st.session_state.br_chat.append({"role": "ai", "content": resp})
                st.rerun()

        if st.session_state.br_chat:
            chat_html = '<div class="chat-container">'
            for msg in st.session_state.br_chat[-6:]:
                chat_html += f'<div class="chat-msg-{"user" if msg["role"]=="user" else "ai"}">{msg["content"]}</div>'
            st.markdown(chat_html + '</div>', unsafe_allow_html=True)

        user_q = st.text_input("Ask...", key="br_user_q", label_visibility="collapsed", placeholder="Tanyakan sesuatu...")
        if user_q:
            st.session_state.br_chat.append({"role": "user", "content": user_q})
            st.session_state.br_chat.append({"role": "ai", "content": f"{n_total} cabang, {n_critical} kritis, avg NPS {avg_nps:.1f}."})
            st.rerun()

        st.markdown(f"""
        <div class="dash-card" style="margin-top:6px;border-top:2px solid {PRIMARY};">
          <div style="font-size:0.7rem;font-weight:700;color:{PRIMARY};margin-bottom:2px">Insight to Action</div>
          <div style="font-size:0.72rem;color:{TEXT_DARK}">Fokus meningkatkan {top_drv_name} di cabang kritis untuk mendongkrak NPS.</div>
        </div>
        """, unsafe_allow_html=True)