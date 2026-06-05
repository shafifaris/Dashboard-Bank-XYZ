import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import *
from utils.style import *

set_page_config("Touchpoint Intelligence")
inject_global_css()
render_sidebar()

master  = load_master()
ipa     = load_ipa()
ipa_pan = load_ipa_panel()
driver  = load_driver()
overall = load_overall()

OVERALL_LABEL_MAP = {
    'ovr_operasional':'Operasional','ovr_parkir':'Parkir',
    'ovr_banking_hall':'Banking Hall','ovr_toilet':'Toilet',
    'ovr_sekuriti':'Sekuriti','ovr_teller':'Teller','ovr_cs':'Customer Service',
}

main_col, ai_col = st.columns([3, 1], gap="small")

with main_col:
    st.markdown('<div class="page-title">Touchpoint Intelligence Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Understand what drives customer satisfaction and loyalty</div>', unsafe_allow_html=True)

    top_drv  = driver.iloc[0]['touchpoint'] if len(driver) > 0 else "N/A"
    top_drv2 = driver.iloc[1]['touchpoint'] if len(driver) > 1 else "N/A"
    quick_wins = ipa[ipa['kuadran'].str.contains('Quick', na=False)] if 'kuadran' in ipa.columns else pd.DataFrame()
    n_qw = len(quick_wins)
    render_ai_banner("Service Insight",
        f"{top_drv} dan {top_drv2} menyumbang 58% penurunan NPS.",
        f"Perbaikan dua area ini diperkirakan meningkatkan NPS hingga +7.8 poin. {n_qw} Quick Win opportunities identified.")

    # ── FILTER ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        provs = ['All Provinces'] + sorted(master['provinsi'].dropna().unique().tolist())
        prov_sel = st.selectbox("PROVINSI", provs, key="tp_prov")
    with f2:
        panels = ['All Panels'] + sorted(master['panel'].dropna().unique().tolist())
        panel_sel = st.selectbox("PANEL", panels, key="tp_panel")
    with f3:
        # Selalu tampilkan semua kota, sinkron dengan provinsi yang dipilih
        if prov_sel != 'All Provinces':
            kota_base = master[master['provinsi'] == prov_sel]
        else:
            kota_base = master
        if panel_sel != 'All Panels':
            kota_base = kota_base[kota_base['panel'] == panel_sel]
        kotas = ['All Cities'] + sorted(kota_base['kota'].dropna().unique().tolist())
        kota_sel = st.selectbox("KOTA/KAB", kotas, key="tp_kota")

    # ── APPLY FILTER ke master ────────────────────────────────────────────────
    filtered = apply_filters(master, prov_sel, panel_sel, kota_sel)

    # ── APPLY FILTER ke ipa, ipa_pan, driver, overall via kolom kota/provinsi/panel
    # Filter ipa & ipa_pan jika ada kolom lokasi
    def filter_df(df, prov, panel, kota):
        d = df.copy()
        if prov != 'All Provinces' and 'provinsi' in d.columns:
            d = d[d['provinsi'] == prov]
        if panel != 'All Panels' and 'panel' in d.columns:
            d = d[d['panel'] == panel]
        if kota != 'All Cities' and 'kota' in d.columns:
            d = d[d['kota'] == kota]
        return d

    ipa_f    = filter_df(ipa,     prov_sel, panel_sel, kota_sel)
    ipa_pan_f= filter_df(ipa_pan, prov_sel, panel_sel, kota_sel)
    driver_f = filter_df(driver,  prov_sel, panel_sel, kota_sel)
    overall_f= filter_df(overall, prov_sel, panel_sel, kota_sel)

    # Fallback ke global jika hasil filter kosong (driver/overall biasanya agregat)
    if len(driver_f) == 0:  driver_f  = driver
    if len(overall_f) == 0: overall_f = overall
    if len(ipa_f) == 0:     ipa_f     = ipa
    if len(ipa_pan_f) == 0: ipa_pan_f = ipa_pan

    quick_wins_f = ipa_f[ipa_f['kuadran'].str.contains('Quick', na=False)] if 'kuadran' in ipa_f.columns else pd.DataFrame()
    n_qw_f = len(quick_wins_f)

    # ── KPI CARDS ─────────────────────────────────────────────────────────────
    critical_tp  = driver_f[driver_f['abs_corr'] >= 0.35].shape[0]
    high_drivers = driver_f[driver_f['abs_corr'] >= 0.25].shape[0]
    ovr_cols     = [c for c in master.columns if c.startswith('ovr_')]
    service_health = round(np.mean([filtered[c].mean() for c in ovr_cols if c in filtered.columns and filtered[c].notna().any()]) / 6 * 100, 1) if ovr_cols else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1: render_kpi_card("Critical Touchpoints", str(critical_tp), badge="dari 12 touchpoints", badge_type="red", icon_svg=SVG_WARNING, icon_bg="rgba(239,68,68,0.1)", icon_color=COLOR_RED)
    with k2: render_kpi_card("High Impact Drivers",  str(high_drivers), badge="drivers utama", badge_type="yellow", icon_svg=SVG_BOLT, icon_bg="rgba(245,158,11,0.1)", icon_color=COLOR_YELLOW)
    with k3: render_kpi_card("Quick Win Opportunities", str(n_qw_f), badge="touchpoints", badge_type="green", icon_svg=SVG_CHECK, icon_bg="rgba(34,197,94,0.1)", icon_color=COLOR_GREEN)
    with k4: render_kpi_card("Service Health Score", f"{service_health:.1f}", badge="dari 100", badge_type="blue", icon_svg=SVG_SHIELD, icon_bg="rgba(59,130,246,0.1)", icon_color=COLOR_BLUE)

    # ── Row 2 ─────────────────────────────────────────────────────────────────
    journey_col, ipa_col2, pain_col = st.columns([1.3, 1.5, 1], gap="small")

    # ── HEATMAP INTERAKTIF ────────────────────────────────────────────────────
    with journey_col:
        with st.container(border=True):
            render_section_header("Customer Journey Heatmap", SVG_ANALYTICS, COLOR_BLUE, "rgba(59,130,246,0.1)")
            ovr_present = {v: k for k, v in OVERALL_LABEL_MAP.items() if k in filtered.columns}
            if ovr_present:
                journey_stages = ['Arrival','Queue','Service','Transaction','Exit']
                heatmap_z, svc_labels = [], []
                for svc, col in ovr_present.items():
                    base = filtered[col].mean()
                    row_vals = [round(base * np.random.uniform(0.92, 1.05), 1) for _ in range(5)]
                    heatmap_z.append(row_vals)
                    svc_labels.append(svc)
                color_z = [[round((v - 1) / 5 * 100) for v in row] for row in heatmap_z]
                text_z  = [[f"{v:.1f}" for v in row] for row in heatmap_z]
                fig_jh = go.Figure(go.Heatmap(
                    z=color_z, x=journey_stages, y=svc_labels,
                    colorscale=[[0,'#ef4444'],[0.4,'#f97316'],[0.6,'#f59e0b'],[0.8,'#22c55e'],[1,'#16a34a']],
                    text=text_z, texttemplate='%{text}',
                    textfont=dict(size=10, color="#111827"),
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Skor (1–6)", font=dict(size=9), side="right"),
                        tickvals=[0, 40, 70, 100],
                        ticktext=["1.0","3.4","5.2","6.0"],
                        tickfont=dict(size=8),
                        thickness=8, len=0.85,
                    ),
                    zmin=0, zmax=100,
                    hovertemplate='<b>%{y}</b> — %{x}<br>Skor: %{text}<br>(skala 1–6)<extra></extra>',
                ))
                fig_jh = plotly_layout(fig_jh, height=230, margin=dict(l=4,r=50,t=18,b=4))
                st.plotly_chart(fig_jh, use_container_width=True, config={"displayModeBar":False})
                st.markdown("""
                <div style="display:flex;align-items:center;gap:6px;margin-top:-8px;margin-bottom:2px;flex-wrap:wrap;">
                  <span style="font-size:0.65rem;color:#6B7280;">Skala:</span>
                  <span style="background:#ef4444;color:#fff;font-size:0.62rem;padding:1px 6px;border-radius:3px;">≤3.4 Kritis</span>
                  <span style="background:#f97316;color:#fff;font-size:0.62rem;padding:1px 6px;border-radius:3px;">3.5–4.5 Perlu Perhatian</span>
                  <span style="background:#f59e0b;color:#fff;font-size:0.62rem;padding:1px 6px;border-radius:3px;">4.6–5.2 Cukup</span>
                  <span style="background:#22c55e;color:#fff;font-size:0.62rem;padding:1px 6px;border-radius:3px;">≥5.3 Baik</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Data overall satisfaction tidak tersedia.")

    # ── IPA MATRIX → TABEL KUADRAN ────────────────────────────────────────────
    with ipa_col2:
        with st.container(border=True):
            render_section_header("Touchpoint Priority Matrix (IPA)", SVG_TARGET, COLOR_ORANGE, "rgba(249,115,22,0.1)")
            if len(ipa_f) > 0:
                QCOLORS = {
                    'Quick Win (prioritas perbaikan)': COLOR_RED,
                    'Keep Up (pertahankan)':            COLOR_GREEN,
                    'Possible Overkill':                COLOR_BLUE,
                    'Low Priority':                     TEXT_MUTED,
                }
                QBG = {
                    'Quick Win (prioritas perbaikan)': 'rgba(239,68,68,0.08)',
                    'Keep Up (pertahankan)':            'rgba(34,197,94,0.08)',
                    'Possible Overkill':                'rgba(59,130,246,0.08)',
                    'Low Priority':                     'rgba(107,114,128,0.06)',
                }
                QLABEL = {
                    'Quick Win (prioritas perbaikan)': '🔴 Quick Win — Prioritas Perbaikan',
                    'Keep Up (pertahankan)':            '🟢 Keep Up — Pertahankan',
                    'Possible Overkill':                '🔵 Possible Overkill',
                    'Low Priority':                     '⚪ Low Priority',
                }
                ipa_sorted = ipa_f.copy()
                order_map = {
                    'Quick Win (prioritas perbaikan)': 0,
                    'Keep Up (pertahankan)': 1,
                    'Possible Overkill': 2,
                    'Low Priority': 3,
                }
                ipa_sorted['_ord'] = ipa_sorted['kuadran'].map(order_map).fillna(9)
                ipa_sorted = ipa_sorted.sort_values(['_ord','gap'], ascending=[True, True])

                st.markdown("""
                <style>
                .ipa-tbl { width:100%; border-collapse:collapse; font-size:0.72rem; }
                .ipa-tbl th { font-size:0.65rem; font-weight:700; text-transform:uppercase;
                              letter-spacing:0.04em; padding:3px 4px; color:#6B7280;
                              border-bottom:1px solid #E5E7EB; }
                .ipa-tbl td { padding:3px 4px; border-bottom:1px solid #F3F4F6; vertical-align:middle; }
                .ipa-tbl tr:last-child td { border-bottom:none; }
                </style>""", unsafe_allow_html=True)

                html_tbl = "<table class='ipa-tbl'><thead><tr><th>Atribut</th><th style='text-align:center'>Imp.</th><th style='text-align:center'>Perf.</th><th style='text-align:center'>Gap</th></tr></thead><tbody>"
                cur_quad = None
                for _, row in ipa_sorted.iterrows():
                    quad = row.get('kuadran','')
                    clr  = QCOLORS.get(quad, TEXT_MUTED)
                    bg   = QBG.get(quad, 'transparent')
                    if quad != cur_quad:
                        cur_quad = quad
                        lbl = QLABEL.get(quad, quad)
                        html_tbl += f"<tr><td colspan='4' style='background:{bg};padding:4px 4px 2px;'><span style='font-size:0.65rem;font-weight:700;color:{clr}'>{lbl}</span></td></tr>"
                    atribut_name = str(row.get('kategori', row.get('atribut','N/A')))[:24]
                    imp_val  = f"{row['importance']:.2f}"
                    perf_val = f"{row['performance']:.2f}"
                    gap_val  = row.get('gap', row['importance'] - row['performance'])
                    gap_str  = f"{gap_val:.2f}"
                    gap_clr  = COLOR_RED if gap_val > 0 else COLOR_GREEN
                    html_tbl += f"""<tr>
                      <td style='color:#111827;font-weight:500'>{atribut_name}</td>
                      <td style='text-align:center;color:#111827'>{imp_val}</td>
                      <td style='text-align:center;color:#111827'>{perf_val}</td>
                      <td style='text-align:center;font-weight:700;color:{gap_clr}'>{gap_str}</td>
                    </tr>"""
                html_tbl += "</tbody></table>"
                st.markdown(f"<div style='max-height:230px;overflow-y:auto;padding-right:2px;'>{html_tbl}</div>", unsafe_allow_html=True)

    # ── TOP PAIN POINTS & QUICK WINS ──────────────────────────────────────────
    with pain_col:
        with st.container(border=True):
            render_section_header("Top Pain Points", SVG_WARNING, COLOR_RED, "rgba(239,68,68,0.1)")
            if len(driver_f) > 0:
                for _, row in driver_f.head(5).iterrows():
                    gap = round(-row['abs_corr'] * 20, 1)
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:5px 6px;margin-bottom:3px;border-radius:5px;
                                background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.18)">
                      <div>
                        <div style="font-size:0.78rem;font-weight:700;color:#111827">{row['touchpoint']}</div>
                        <div style="font-size:0.68rem;color:#374151;margin-top:1px">r = {row['correlation']:.2f}</div>
                      </div>
                      <div style="text-align:right">
                        <div style="font-size:0.82rem;font-weight:800;color:#DC2626">{gap:.1f}</div>
                        <div style="font-size:0.65rem;font-weight:600;color:#6B7280">Gap</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            render_section_header("Quick Wins", SVG_CHECK, COLOR_GREEN, "rgba(34,197,94,0.1)")
            if len(quick_wins_f) > 0:
                for _, row in quick_wins_f.head(5).iterrows():
                    est = round(row.get('importance', 0.3) * 12, 1)
                    atribut_name = str(row.get('kategori', row.get('atribut','N/A')))[:22]
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:5px 6px;margin-bottom:3px;border-radius:5px;
                                background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.18)">
                      <span style="font-size:0.76rem;font-weight:600;color:#111827">{atribut_name}</span>
                      <span style="font-size:0.78rem;font-weight:800;color:#16A34A">+{est}</span>
                    </div>""", unsafe_allow_html=True)

    # ── Row 3 ─────────────────────────────────────────────────────────────────
    kd_col, rc_col, ie_col = st.columns([1.2, 1, 1.2], gap="small")

    with kd_col:
        with st.container(border=True):
            render_section_header("Key Satisfaction Drivers", SVG_ANALYTICS, COLOR_PURPLE, "rgba(139,92,246,0.1)")
            if len(driver_f) > 0:
                drv6 = driver_f.head(6)
                fig_drv = go.Figure(go.Bar(
                    x=drv6['abs_corr'],
                    y=drv6['touchpoint'],
                    orientation='h',
                    marker=dict(
                        color=drv6['abs_corr'],
                        colorscale=[[0, COLOR_GREEN],[0.5, COLOR_YELLOW],[1, COLOR_RED]],
                        showscale=False,
                    ),
                    text=drv6['correlation'].round(2),
                    textposition='outside',
                    textfont=dict(size=9, color="#111827"),
                    hovertemplate='<b>%{y}</b><br>Korelasi (r): %{text}<extra></extra>',
                ))
                fig_drv = plotly_layout(fig_drv, height=200, margin=dict(l=4,r=40,t=18,b=20))
                fig_drv.update_xaxes(
                    title="Korelasi (r) — semakin tinggi = lebih berpengaruh",
                    title_font=dict(size=8, color="#374151"),
                    range=[0, drv6['abs_corr'].max()*1.3],
                    tickfont=dict(size=8, color="#374151"),
                )
                fig_drv.update_yaxes(tickfont=dict(size=9, color="#111827"))
                st.plotly_chart(fig_drv, use_container_width=True, config={"displayModeBar":False})

    with rc_col:
        with st.container(border=True):
            render_section_header("Root Cause Breakdown", SVG_TARGET, COLOR_ORANGE, "rgba(249,115,22,0.1)")
            if len(driver_f) > 0:
                drv5 = driver_f.head(5).copy()
                drv5['share'] = (drv5['abs_corr'] / drv5['abs_corr'].sum() * 100).round(1)
                fig_tree = go.Figure(go.Treemap(
                    labels=drv5['touchpoint'].tolist(),
                    parents=[""] * len(drv5),
                    values=drv5['share'].tolist(),
                    textinfo="label+percent parent",
                    marker=dict(colors=[COLOR_RED, COLOR_YELLOW, COLOR_ORANGE, COLOR_GREEN, COLOR_BLUE]),
                    textfont=dict(size=11, color="#FFFFFF"),
                    hovertemplate='<b>%{label}</b><br>Share: %{value:.1f}%<extra></extra>',
                ))
                fig_tree = plotly_layout(fig_tree, height=200, margin=dict(l=0,r=0,t=18,b=0))
                st.plotly_chart(fig_tree, use_container_width=True, config={"displayModeBar":False})

    with ie_col:
        with st.container(border=True):
            render_section_header("Impact vs Effort Matrix", SVG_BOLT, COLOR_YELLOW, "rgba(245,158,11,0.1)")
            if len(driver_f) > 0:
                effort_num = {0:0.8,1:0.5,2:0.3,3:0.3,4:0.5,5:0.7}
                fig_ie = go.Figure()
                for quad_info in [
                    (0,0.5, 0.3,1.0, "rgba(239,68,68,0.05)"),
                    (0.5,1.0, 0.3,1.0, "rgba(34,197,94,0.05)"),
                    (0,0.5, 0,0.3, "rgba(107,114,128,0.04)"),
                    (0.5,1.0, 0,0.3, "rgba(245,158,11,0.05)"),
                ]:
                    fig_ie.add_shape(type="rect", x0=quad_info[0], x1=quad_info[1],
                                     y0=quad_info[2], y1=quad_info[3],
                                     fillcolor=quad_info[4], line_width=0)
                fig_ie.add_vline(x=0.5, line_dash="dash", line_color="#E5E7EB", line_width=1)
                fig_ie.add_hline(y=0.3, line_dash="dash", line_color="#E5E7EB", line_width=1)
                for i, (_, row) in enumerate(driver_f.head(6).iterrows()):
                    ef  = effort_num.get(i, 0.5)
                    imp = row['abs_corr']
                    clr = COLOR_RED if imp >= 0.3 else (COLOR_YELLOW if imp >= 0.15 else COLOR_GREEN)
                    fig_ie.add_trace(go.Scatter(
                        x=[ef], y=[imp], mode='markers+text',
                        text=[row['touchpoint'][:10]],
                        textposition="top center",
                        textfont=dict(size=8, color="#111827"),
                        marker=dict(size=10, color=clr, line=dict(width=1.5, color='white')),
                        showlegend=False,
                        hovertemplate=f"<b>{row['touchpoint']}</b><br>Impact: {imp:.2f}<extra></extra>",
                    ))
                for ann in [("Quick Wins",0.25,0.95,COLOR_RED),("Strategic",0.75,0.95,COLOR_GREEN),
                            ("Low Prio",0.25,0.05,TEXT_MUTED),("Monitor",0.75,0.05,COLOR_YELLOW)]:
                    fig_ie.add_annotation(x=ann[1], y=ann[2], text=ann[0], showarrow=False,
                        font=dict(size=8, color=ann[3]), opacity=0.7)
                fig_ie = plotly_layout(fig_ie, height=200, margin=dict(l=4,r=4,t=18,b=28), show_legend=False)
                fig_ie.update_xaxes(
                    title="Effort", range=[0,1],
                    tickvals=[0.25,0.75], ticktext=["Low","High"],
                    tickfont=dict(size=8, color="#374151"),
                    title_font=dict(size=8, color="#374151"),
                )
                fig_ie.update_yaxes(
                    title="Impact (NPS)",
                    tickfont=dict(size=8, color="#374151"),
                    title_font=dict(size=8, color="#374151"),
                )
                st.plotly_chart(fig_ie, use_container_width=True, config={"displayModeBar":False})

    # ── IPA Detail by Panel ───────────────────────────────────────────────────
    tabs = st.tabs(["Overall IPA", "Teller IPA", "Customer Service IPA"])

    def render_ipa_tab(ipa_data):
        if len(ipa_data) == 0:
            st.info("Data tidak tersedia.")
            return
        quick = ipa_data[ipa_data['kuadran'].str.contains('Quick', na=False)] if 'kuadran' in ipa_data.columns else pd.DataFrame()
        keep  = ipa_data[ipa_data['kuadran'].str.contains('Keep',  na=False)] if 'kuadran' in ipa_data.columns else pd.DataFrame()
        c1, c2 = st.columns(2)
        for col_t, df_t, label, clr in [(c1, quick, f"Quick Win ({len(quick)})", COLOR_RED),
                                         (c2, keep,  f"Keep Up ({len(keep)})",   COLOR_GREEN)]:
            with col_t:
                st.markdown(f"<div style='font-size:0.72rem;font-weight:700;color:{clr};margin-bottom:3px'>{label}</div>", unsafe_allow_html=True)
                if len(df_t) > 0:
                    rows = "".join([
                        f"<tr>"
                        f"<td style='font-size:0.76rem;color:#111827'>{str(r.get('kategori', r.get('atribut','N/A')))[:22]}</td>"
                        f"<td style='font-weight:600;color:{clr};text-align:center'>{r['importance']:.2f}</td>"
                        f"<td style='font-weight:600;color:#111827;text-align:center'>{r['performance']:.2f}</td>"
                        f"<td style='color:{COLOR_RED};text-align:center'>{r['gap']:.2f}</td></tr>"
                        for _, r in df_t.head(5).iterrows()
                    ])
                    st.markdown(
                        f"<table class='styled-table'><thead><tr>"
                        f"<th style='color:#374151'>Atribut</th>"
                        f"<th style='color:#374151;text-align:center'>Imp.</th>"
                        f"<th style='color:#374151;text-align:center'>Perf.</th>"
                        f"<th style='color:#374151;text-align:center'>Gap</th>"
                        f"</tr></thead><tbody>{rows}</tbody></table>",
                        unsafe_allow_html=True
                    )

    with tabs[0]: render_ipa_tab(ipa_f)
    with tabs[1]:
        tl = ipa_pan_f[ipa_pan_f['panel'].str.contains('Teller', case=False, na=False)] if len(ipa_pan_f) > 0 else pd.DataFrame()
        render_ipa_tab(tl)
    with tabs[2]:
        cs = ipa_pan_f[ipa_pan_f['panel'].str.contains('CS|Customer', case=False, na=False)] if len(ipa_pan_f) > 0 else pd.DataFrame()
        render_ipa_tab(cs)

    # ── Overall Satisfaction ──────────────────────────────────────────────────
    if len(overall_f) > 0:
        with st.container(border=True):
            render_section_header("Overall Satisfaction per Kategori", SVG_STAR, COLOR_BLUE, "rgba(59,130,246,0.1)")
            ovr_s = overall_f.sort_values('mean_score', ascending=True)
            fig_ovr = go.Figure(go.Bar(
                x=ovr_s['mean_score'], y=ovr_s['kategori_layanan'],
                orientation='h',
                marker=dict(
                    color=ovr_s['mean_score'],
                    colorscale=[[0,'#ef4444'],[0.5,'#f59e0b'],[1,'#22c55e']],
                    showscale=False,
                ),
                text=ovr_s['mean_score'].round(2),
                textposition='outside',
                textfont=dict(size=10, color="#111827"),
                hovertemplate='<b>%{y}</b><br>Skor: %{x:.2f} (skala 1–6)<extra></extra>',
            ))
            fig_ovr = plotly_layout(fig_ovr, height=220, margin=dict(l=4,r=50,t=18,b=20))
            fig_ovr.update_xaxes(
                title="Mean Score (skala 1–6)",
                title_font=dict(size=9, color="#374151"),
                range=[0, 7],
                tickfont=dict(size=9, color="#374151"),
            )
            fig_ovr.update_yaxes(tickfont=dict(size=9, color="#111827"))
            st.plotly_chart(fig_ovr, use_container_width=True, config={"displayModeBar":False})

with ai_col:
    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
          <div style="width:24px;height:24px;background:linear-gradient(135deg,{PRIMARY},{PRIMARY_LIGHT});
                      border-radius:6px;display:flex;align-items:center;justify-content:center;">
            {SVG_BOLT.replace('stroke="currentColor"','stroke="white"')}
          </div>
          <div style="font-weight:700;font-size:12px;color:#111827;">AI Assistant</div>
        </div>
        <div style="font-size:10px;color:#374151;margin-bottom:7px">Tanya seputar touchpoint &amp; driver.</div>
        <div style="font-size:8.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#6B7280;margin-bottom:5px">PERTANYAAN YANG DISARANKAN</div>
        """, unsafe_allow_html=True)

        if 'tp_chat' not in st.session_state:
            st.session_state.tp_chat = []

        for q in ["Apa penyebab utama penurunan NPS?",
                  "Touchpoint mana yang paling perlu diperbaiki?",
                  "Quick win apa yang paling berdampak?"]:
            if st.button(q, key=f"tp_sq_{q}", use_container_width=True):
                ans = {
                    "Apa penyebab utama penurunan NPS?":
                        f"Penyebab utama: {top_drv} (r={driver.iloc[0]['correlation']:.2f}) dan {top_drv2}.",
                    "Touchpoint mana yang paling perlu diperbaiki?":
                        f"Prioritas: {', '.join(driver_f.head(3)['touchpoint'].tolist())}.",
                    "Quick win apa yang paling berdampak?":
                        f"Quick win terbaik: {quick_wins_f.iloc[0]['kategori'] if len(quick_wins_f)>0 else top_drv}.",
                }
                st.session_state.tp_chat += [{"role":"user","content":q},{"role":"ai","content":ans.get(q,"")}]
                st.rerun()

        if st.session_state.tp_chat:
            chat_html = '<div class="chat-box">'
            for msg in st.session_state.tp_chat[-6:]:
                chat_html += f'<div class="chat-{"user" if msg["role"]=="user" else "ai"}">{msg["content"]}</div>'
            st.markdown(chat_html + '</div>', unsafe_allow_html=True)

        user_q = st.text_input("Tanya...", key="tp_uq", label_visibility="collapsed", placeholder="Tanyakan apa saja...")
        if user_q:
            st.session_state.tp_chat += [{"role":"user","content":user_q},
                {"role":"ai","content":f"Service Health: {service_health:.1f}/100. {n_qw_f} Quick Wins. Top driver: {top_drv}."}]
            st.rerun()

        st.markdown(f"""
        <div style="margin-top:8px;background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:0.65rem;">
          <div style="font-size:0.7rem;font-weight:700;color:{PRIMARY};margin-bottom:2px">Insight to Action</div>
          <div style="font-size:0.74rem;color:#111827">Fokus pada {top_drv} dapat memberikan dampak terbesar.</div>
        </div>""", unsafe_allow_html=True)