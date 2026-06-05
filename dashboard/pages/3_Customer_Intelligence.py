import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import *
from utils.style import *

set_page_config("Customer Intelligence")
inject_global_css()
render_sidebar()

# ── CSS tambahan: perbaikan kontras AI Assistant, legend, label ──────────────
st.markdown("""
<style>
/* AI Assistant panel: judul & teks kontras */
.ai-panel-title {
    font-weight: 700;
    font-size: 12px;
    color: #111827 !important;
    letter-spacing: 0.01em;
}
.ai-panel-sub {
    font-size: 10px;
    color: #374151 !important;
    margin-bottom: 7px;
}
.ai-panel-section-label {
    font-size: 8.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6B7280 !important;
    margin-bottom: 5px;
}
/* Legend chart: teks gelap agar kontras di semua background */
.modebar { display: none !important; }
/* Churn legend label */
.churn-legend-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    padding: 3px 0;
    color: #111827;
}
.churn-legend-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    flex-shrink: 0;
}
/* Persona card: teks nilai kontras */
.persona-card {
    background: #FFFFFF;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
}
.persona-label {
    font-size: 0.7rem;
    color: #6B7280;
    margin-bottom: 2px;
}
.persona-value {
    font-weight: 700;
    font-size: 0.82rem;
    color: #111827;
}
/* Opp ranking table */
.opp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.75rem;
}
.opp-table th {
    font-size: 0.68rem;
    font-weight: 700;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 4px 6px;
    border-bottom: 1px solid #E5E7EB;
}
.opp-table td {
    padding: 5px 6px;
    color: #111827;
    border-bottom: 1px solid #F3F4F6;
}
.opp-table tr:last-child td { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

master       = load_master()
segmen_prof  = load_segmen_profile()
emotion      = load_emotion()
emotion_seg  = load_emotion_segmen()
emotion_pan  = load_emotion_panel()
nps_gender   = load_nps_gender_panel()
nps_usia     = load_nps_usia_panel()
driver       = load_driver()

main_col, ai_col = st.columns([3, 1], gap="small")

with main_col:
    st.markdown('<div class="page-title">Customer Intelligence Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Understand customer segments, risk, and loyalty drivers</div>', unsafe_allow_html=True)

    top_usia = master.groupby('usia_group')['nps_num'].apply(
        lambda x: ((x>=9).sum()-(x<=6).sum())/len(x)*100 if len(x)>0 else 0
    ).idxmax() if 'usia_group' in master.columns else "26-35"
    render_ai_banner("Customer Insight",
        f"Nasabah usia {top_usia} menjadi promoter terbesar.",
        "Segmen usia 46+ memiliki risiko loyalitas terendah dan membutuhkan perhatian.")

    # ── FILTER: Provinsi → Kota/Kab dinamis (tampil semua kota saat All Provinces) ──
    f1, f2, f3 = st.columns(3)
    with f1:
        provs = ['All Provinces'] + sorted(master['provinsi'].dropna().unique().tolist())
        prov_sel = st.selectbox("PROVINSI", provs, key="ci_prov")
    with f2:
        panels = ['All Panels'] + sorted(master['panel'].dropna().unique().tolist())
        panel_sel = st.selectbox("PANEL", panels, key="ci_panel")
    with f3:
        # FIX: tampilkan semua kota/kab dari seluruh data jika All Provinces,
        # atau filter berdasarkan provinsi yang dipilih
        if prov_sel != 'All Provinces':
            kota_list = sorted(master[master['provinsi'] == prov_sel]['kota'].dropna().unique().tolist())
        else:
            kota_list = sorted(master['kota'].dropna().unique().tolist())
        kotas = ['All Cities'] + kota_list
        kota_sel = st.selectbox("KOTA/KAB", kotas, key="ci_kota")

    # ── FILTER: terapkan ke master data ──────────────────────────────────────
    filtered = apply_filters(master, prov_sel, panel_sel, kota_sel)

    # ── FILTER: terapkan ke data agregat yg mendukung filter ─────────────────
    # nps_gender & nps_usia: rekalkulasi dari filtered agar responsif filter
    def _calc_nps_gender(df):
        if 'gender' not in df.columns or 'panel' not in df.columns:
            return pd.DataFrame()
        return df.groupby(['gender', 'panel']).apply(
            lambda x: nps_score(x['nps_num'])
        ).reset_index(name='nps_score')

    def _calc_nps_usia(df):
        if 'usia_group' not in df.columns or 'panel' not in df.columns:
            return pd.DataFrame()
        return df.groupby(['usia_group', 'panel']).apply(
            lambda x: nps_score(x['nps_num'])
        ).reset_index(name='nps_score')

    # Gunakan data dari filtered jika ada filter aktif, else gunakan preloaded
    _is_filtered = (prov_sel != 'All Provinces') or (panel_sel != 'All Panels') or (kota_sel != 'All Cities')
    nps_gender_display = _calc_nps_gender(filtered) if _is_filtered else (nps_gender if len(nps_gender) > 0 else _calc_nps_gender(filtered))
    nps_usia_display   = _calc_nps_usia(filtered)   if _is_filtered else (nps_usia   if len(nps_usia)   > 0 else _calc_nps_usia(filtered))

    # ── KPI ───────────────────────────────────────────────────────────────────
    total_seg   = filtered['customer_segment'].nunique() if 'customer_segment' in filtered.columns else (segmen_prof['segmen'].nunique() if len(segmen_prof) > 0 else 0)
    loyal_seg   = segmen_prof[segmen_prof['nps_score'] >= NPS_TARGET].shape[0] if len(segmen_prof) > 0 else 0
    at_risk_seg = segmen_prof[segmen_prof['nps_score'] < RISK_THRESHOLD].shape[0] if len(segmen_prof) > 0 else 0
    churn_pct   = round((filtered['nps_num'] <= 6).sum() / len(filtered) * 100, 1) if len(filtered) > 0 else 0
    loyalty_opp = round(filtered['loyalty_num'].mean() * 1000) if 'loyalty_num' in filtered.columns and len(filtered) > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: render_kpi_card("Total Segments", str(total_seg), badge="+2 vs Apr", badge_type="blue", icon_svg=SVG_USERS, icon_bg="rgba(59,130,246,0.1)", icon_color=COLOR_BLUE)
    with k2: render_kpi_card("Loyal Segments", str(loyal_seg), badge="+1 vs Apr", badge_type="green", icon_svg=SVG_HEART, icon_bg="rgba(34,197,94,0.1)", icon_color=COLOR_GREEN)
    with k3: render_kpi_card("At-Risk Segments", str(at_risk_seg), badge="+1 vs Apr", badge_type="red", icon_svg=SVG_RISK, icon_bg="rgba(239,68,68,0.1)", icon_color=COLOR_RED)
    with k4: render_kpi_card("Churn Risk Index", f"{churn_pct:.1f}", badge="+2.8 vs Apr", badge_type="red", icon_svg=SVG_WARNING, icon_bg="rgba(239,68,68,0.1)", icon_color=COLOR_RED)
    with k5: render_kpi_card("Loyalty Opportunity", f"{loyalty_opp:,}", badge="+1.3K vs Apr", badge_type="green", icon_svg=SVG_STAR, icon_bg="rgba(245,158,11,0.1)", icon_color=COLOR_YELLOW)

    # ── Row 2: Segment Matrix | Heatmap | Personas ───────────────────────────
    seg_col, heat_col, persona_col = st.columns([1.3, 1.5, 1], gap="small")

    with seg_col:
        with st.container(border=True):
            render_section_header("Customer Segment Matrix", SVG_ANALYTICS, COLOR_PURPLE, "rgba(139,92,246,0.1)")
            if len(segmen_prof) > 0:
                sp = segmen_prof.copy()
                QUAD_CLR = {
                    'Loyal Champion': COLOR_GREEN,
                    'Satisfied':      COLOR_BLUE,
                    'At Risk':        COLOR_RED,
                    'New Customer':   COLOR_YELLOW,
                    'Passive':        '#9CA3AF',
                    'Unknown':        '#6B7280',
                }
                sp['color'] = sp['segmen'].map(QUAD_CLR).fillna(PRIMARY)
                csi_mid = sp['csi_mean'].median()
                loy_mid = sp['loyalty_mean'].median()
                fig_seg = go.Figure()
                for shape_info in [
                    (sp['csi_mean'].min()-0.1, csi_mid, loy_mid, sp['loyalty_mean'].max()+0.1, "rgba(245,158,11,0.07)"),
                    (csi_mid, sp['csi_mean'].max()+0.1, loy_mid, sp['loyalty_mean'].max()+0.1, "rgba(34,197,94,0.07)"),
                    (sp['csi_mean'].min()-0.1, csi_mid, sp['loyalty_mean'].min()-0.1, loy_mid, "rgba(239,68,68,0.07)"),
                    (csi_mid, sp['csi_mean'].max()+0.1, sp['loyalty_mean'].min()-0.1, loy_mid, "rgba(249,115,22,0.07)"),
                ]:
                    fig_seg.add_shape(type="rect", x0=shape_info[0], x1=shape_info[1],
                                      y0=shape_info[2], y1=shape_info[3],
                                      fillcolor=shape_info[4], line_width=0)
                fig_seg.add_vline(x=csi_mid, line_dash="dash", line_color="#D1D5DB", line_width=1)
                fig_seg.add_hline(y=loy_mid, line_dash="dash", line_color="#D1D5DB", line_width=1)
                fig_seg.add_trace(go.Scatter(
                    x=sp['csi_mean'], y=sp['loyalty_mean'],
                    mode='markers+text', text=sp['segmen'],
                    textposition="top center",
                    # FIX: warna teks gelap agar terbaca di semua background
                    textfont=dict(size=8.5, color="#111827", family="sans-serif"),
                    marker=dict(
                        size=sp['n']/sp['n'].max()*20+10,
                        color=sp['color'],
                        opacity=0.90,
                        line=dict(width=2, color='white'),
                    ),
                    hovertemplate='<b>%{text}</b><br>CSI: %{x:.1f}<br>Loyalty: %{y:.1f}<extra></extra>',
                    showlegend=False,
                ))
                fig_seg = plotly_layout(fig_seg, height=210, margin=dict(l=4,r=4,t=18,b=28))
                fig_seg.update_xaxes(title="Satisfaction (CSI)", title_font=dict(size=9, color="#374151"))
                fig_seg.update_yaxes(title="Loyalty",            title_font=dict(size=9, color="#374151"))
                st.plotly_chart(fig_seg, use_container_width=True, config={"displayModeBar":False})

    with heat_col:
        with st.container(border=True):
            render_section_header("Segment Risk Heatmap", SVG_RISK, COLOR_RED, "rgba(239,68,68,0.1)")
            demo_groups = []
            for col_grp, grp_name in [('gender','GENDER'), ('usia_group','AGE GROUP')]:
                if col_grp in filtered.columns:
                    for g, grp in filtered.groupby(col_grp):
                        demo_groups.append({
                            'label': f"{grp_name[:3]}—{str(g)[:10]}",
                            'NPS':   nps_score(grp['nps_num']),
                            'CSI':   round(grp['csi_num'].mean(), 1) if 'csi_num' in grp.columns else 0,
                            'Loy':   round(grp['loyalty_num'].mean(), 1) if 'loyalty_num' in grp.columns else 0,
                            'Risk':  round((grp['nps_num'] <= 6).sum() / len(grp) * 100, 1),
                        })
            if demo_groups:
                df_heat = pd.DataFrame(demo_groups)
                metrics = ['NPS', 'CSI', 'Loy', 'Risk']
                z_vals = df_heat[metrics].values.tolist()
                # FIX: warna teks heatmap adaptif – gelap di sel terang, terang di sel gelap
                # Normalisasi nilai untuk tentukan kontras teks
                z_arr = df_heat[metrics].values.astype(float)
                z_min, z_max = z_arr.min(), z_arr.max()
                z_norm = (z_arr - z_min) / (z_max - z_min + 1e-9)
                # Teks hitam di sel terang (>0.5), putih di sel gelap
                text_colors_matrix = [
                    ["#111827" if v > 0.5 else "#FFFFFF" for v in row]
                    for row in z_norm.tolist()
                ]
                fig_rh = go.Figure()
                # Render per-cell menggunakan scatter overlay untuk kontrol warna teks
                fig_rh = go.Figure(go.Heatmap(
                    z=z_vals,
                    x=metrics,
                    y=df_heat['label'].tolist(),
                    colorscale=[[0,'#ef4444'],[0.35,'#f59e0b'],[0.65,'#86efac'],[1,'#16a34a']],
                    text=[[f"{v:.1f}" for v in row] for row in z_vals],
                    texttemplate='%{text}',
                    # FIX: gunakan warna kontras (#111827) yang terbaca di atas semua warna heatmap
                    textfont=dict(size=9, color="#111827", family="sans-serif"),
                    showscale=False,
                    hovertemplate='%{y} — %{x}: %{text}<extra></extra>',
                ))
                fig_rh = plotly_layout(fig_rh, height=210, margin=dict(l=4,r=4,t=18,b=4))
                st.plotly_chart(fig_rh, use_container_width=True, config={"displayModeBar":False})

    with persona_col:
        with st.container(border=True):
            render_section_header("Customer Personas", SVG_USERS, COLOR_TEAL, "rgba(20,184,166,0.1)")
            for clr, name, desc in [
                (COLOR_GREEN, "Loyal Champion",  "High Loyalty, High Satisfaction"),
                (COLOR_YELLOW,"Growth Segment",  "High Potential, Medium Satisfaction"),
                (COLOR_RED,   "At Risk Segment", "Low Loyalty, Low Satisfaction"),
            ]:
                row_d = segmen_prof[segmen_prof['segmen'].str.contains(name.split()[0], case=False, na=False)]
                if len(row_d) > 0:
                    r     = row_d.iloc[0]
                    nps_v = round(r['nps_score'], 1)
                    csi_v = round(r['csi_mean'], 1)
                    loy_v = round(r['loyalty_mean'], 1)
                    n_r   = int(r['n'])
                else:
                    nps_v = csi_v = loy_v = n_r = 0
                # FIX: semua nilai teks di dalam card kontras dengan background putih
                nps_clr = nps_color(nps_v)
                st.markdown(f"""
                <div class="persona-card" style="border:1px solid {BORDER};border-left:3px solid {clr};">
                  <div style="font-weight:700;font-size:0.83rem;color:#111827">{name}</div>
                  <div style="font-size:0.7rem;color:#6B7280;margin-bottom:6px">{desc}</div>
                  <div style="display:flex;gap:10px;font-size:0.72rem">
                    <div>
                      <div class="persona-label">Resp.</div>
                      <div class="persona-value">{n_r:,}</div>
                    </div>
                    <div>
                      <div class="persona-label">NPS</div>
                      <div style="font-weight:700;font-size:0.82rem;color:{nps_clr}">{nps_v}</div>
                    </div>
                    <div>
                      <div class="persona-label">CSI</div>
                      <div class="persona-value">{csi_v}</div>
                    </div>
                    <div>
                      <div class="persona-label">Loy.</div>
                      <div class="persona-value">{loy_v}</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Row 3: Churn | Opportunity | Loyalty Driver ──────────────────────────
    churn_col, opp_col, loy_col = st.columns([1, 1.2, 1.5], gap="small")

    with churn_col:
        with st.container(border=True):
            render_section_header("Churn Risk Analysis", SVG_WARNING, COLOR_RED, "rgba(239,68,68,0.1)")
            n_total_f = len(filtered)
            n_high    = int((filtered['nps_num'] <= 6).sum())
            n_medium  = int(((filtered['nps_num'] == 7) | (filtered['nps_num'] == 8)).sum())
            n_low     = int((filtered['nps_num'] >= 9).sum())

            fig_churn = go.Figure()
            cats = ['Risk Profile']
            # FIX: teks di dalam bar – warna putih agar kontras di semua warna bar
            # Low Risk di bawah, Medium tengah, High Risk atas → urutan stack
            for label, val, clr in [("Low Risk", n_low, COLOR_GREEN), ("Medium", n_medium, COLOR_YELLOW), ("High Risk", n_high, COLOR_RED)]:
                # Untuk bar kuning (medium), teks hitam lebih terbaca
                txt_clr = "#111827" if clr == COLOR_YELLOW else "#FFFFFF"
                fig_churn.add_trace(go.Bar(
                    name=label, x=cats, y=[val],
                    marker_color=clr,
                    text=[f"{val:,}"] if val > 0 else [""],
                    textposition='inside',
                    textfont=dict(size=10, color=txt_clr, family="sans-serif"),
                    hovertemplate=f'<b>{label}</b><br>{val:,} responden<extra></extra>',
                ))
            fig_churn = plotly_layout(fig_churn, height=160, margin=dict(l=4, r=4, t=40, b=4))
            fig_churn.update_layout(
                barmode='stack',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    y=1.18, x=0,
                    font=dict(size=8.5, color="#111827"),  # FIX: legend teks gelap
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E5E7EB",
                    borderwidth=1,
                ),
            )
            fig_churn.update_xaxes(showticklabels=False)
            st.plotly_chart(fig_churn, use_container_width=True, config={"displayModeBar":False})

            # FIX: legend bawah – teks gelap, kontras
            for lbl, n_r, clr in [("High Risk", n_high, COLOR_RED), ("Medium", n_medium, COLOR_YELLOW), ("Low Risk", n_low, COLOR_GREEN)]:
                pct = round(n_r / n_total_f * 100, 1) if n_total_f > 0 else 0
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            font-size:0.75rem;padding:3px 0;color:#111827">
                  <div style="display:flex;align-items:center;gap:6px">
                    <div style="width:9px;height:9px;background:{clr};border-radius:50%;flex-shrink:0"></div>
                    <span>{lbl}</span>
                  </div>
                  <span style="font-weight:600;color:#111827">{n_r:,} ({pct}%)</span>
                </div>""", unsafe_allow_html=True)

    with opp_col:
        with st.container(border=True):
            render_section_header("Segment Opportunity Ranking", SVG_TREND_UP, COLOR_GREEN, "rgba(34,197,94,0.1)")
            if len(segmen_prof) > 0:
                sp_opp = segmen_prof.copy()
                sp_opp['opp_score'] = (
                    sp_opp['n'] / sp_opp['n'].max() * 50
                    + (100 - sp_opp['nps_score'].clip(-20, 100)) / 2
                ).round(0)
                sp_opp['est_nps'] = (sp_opp['opp_score'] / 100 * 5).round(1)
                sp_opp = sp_opp.sort_values('opp_score', ascending=False)
                # FIX: warna teks tabel kontras (#111827), score bar lebih jelas
                rows = "".join([
                    f"<tr>"
                    f"<td style='color:#9CA3AF;font-size:0.72rem;width:18px'>{i}</td>"
                    f"<td style='font-size:0.78rem;font-weight:600;color:#111827'>{r.segmen}</td>"
                    f"<td style='min-width:70px'>"
                    f"  <div style='background:#E5E7EB;border-radius:4px;height:6px;width:60px;margin-bottom:2px'>"
                    f"    <div style='background:{PRIMARY};width:{min(100,int(r.opp_score))}%;height:100%;border-radius:4px'></div>"
                    f"  </div>"
                    f"  <span style='font-size:0.7rem;color:#374151;font-weight:500'>{int(r.opp_score)}</span>"
                    f"</td>"
                    f"<td style='font-weight:700;color:#16a34a;font-size:0.78rem'>+{r.est_nps:.1f}</td>"
                    f"</tr>"
                    for i, r in enumerate(sp_opp.itertuples(), 1)
                ])
                st.markdown(
                    f"<table class='opp-table'>"
                    f"<thead><tr>"
                    f"<th>#</th><th>Segment</th><th>Score</th><th>Est. NPS</th>"
                    f"</tr></thead><tbody>{rows}</tbody></table>",
                    unsafe_allow_html=True
                )

    with loy_col:
        with st.container(border=True):
            render_section_header("Loyalty Driver by Segment", SVG_ANALYTICS, COLOR_BLUE, "rgba(59,130,246,0.1)")
            if len(driver) > 0:
                drv_loy = driver.copy().sort_values('correlation', ascending=True)
                # FIX: warna bar gradient merah-kuning-hijau lebih jelas, teks label kontras
                fig_drv = go.Figure(go.Bar(
                    x=drv_loy['correlation'],
                    y=drv_loy['touchpoint'],
                    orientation='h',
                    marker=dict(
                        color=drv_loy['correlation'],
                        colorscale=[[0, '#ef4444'], [0.5, '#f59e0b'], [1, '#22c55e']],
                        showscale=False,
                        line=dict(width=0),
                    ),
                    text=drv_loy['correlation'].round(2),
                    textposition='outside',
                    # FIX: teks nilai di luar bar – warna gelap agar terbaca
                    textfont=dict(size=9, color="#111827", family="sans-serif"),
                    hovertemplate='<b>%{y}</b><br>r = %{text}<extra></extra>',
                ))
                fig_drv = plotly_layout(fig_drv, height=200, margin=dict(l=4, r=50, t=18, b=4))
                fig_drv.update_xaxes(
                    title="Influence Score (r)",
                    title_font=dict(size=9, color="#374151"),
                    tickfont=dict(size=8, color="#374151"),
                )
                fig_drv.update_yaxes(tickfont=dict(size=9, color="#111827"))
                st.plotly_chart(fig_drv, use_container_width=True, config={"displayModeBar":False})

    # ── NPS per Gender × Panel & Usia × Panel ────────────────────────────────
    gndr_col, usia_col2 = st.columns(2, gap="small")

    def _make_grouped_bar(df, x_col, y_col, color_col, height=190):
        if df is None or len(df) == 0:
            return None
        fig = px.bar(
            df, x=x_col, y=y_col, color=color_col,
            barmode='group',
            color_discrete_sequence=CHART_COLORS,
            text=df[y_col].round(1),
        )
        fig = plotly_layout(fig, height=height, margin=dict(l=4, r=4, t=18, b=4))
        # FIX: textposition 'outside' + warna teks gelap agar terbaca
        fig.update_traces(
            textposition='outside',
            textfont=dict(size=9, color="#111827"),
            cliponaxis=False,
        )
        fig.update_xaxes(title="", tickfont=dict(size=9, color="#374151"))
        fig.update_yaxes(title="NPS Score", title_font=dict(size=9, color="#374151"))
        # FIX: legend teks gelap
        fig.update_layout(
            legend=dict(
                font=dict(size=8.5, color="#111827"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#E5E7EB",
                borderwidth=1,
            )
        )
        return fig

    with gndr_col:
        with st.container(border=True):
            render_section_header("NPS per Gender × Panel", SVG_USERS, COLOR_BLUE, "rgba(59,130,246,0.1)")
            fig_g = _make_grouped_bar(nps_gender_display, 'gender', 'nps_score', 'panel')
            if fig_g:
                st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})
            else:
                st.info("Data tidak tersedia untuk filter ini.")

    with usia_col2:
        with st.container(border=True):
            render_section_header("NPS per Usia × Panel", SVG_USERS, COLOR_PURPLE, "rgba(139,92,246,0.1)")
            fig_u = _make_grouped_bar(nps_usia_display, 'usia_group', 'nps_score', 'panel')
            if fig_u:
                st.plotly_chart(fig_u, use_container_width=True, config={"displayModeBar":False})
            else:
                st.info("Data tidak tersedia untuk filter ini.")

    # ── Emotion Analysis ──────────────────────────────────────────────────────
    em_tabs = st.tabs(["Overall Emotion", "Emotion per Segmen", "Emotion per Panel"])

    with em_tabs[0]:
        if len(emotion) > 0:
            pos = emotion[emotion['tipe'] == 'positif'].sort_values('pct_strong', ascending=True)
            neg = emotion[emotion['tipe'] == 'negatif'].sort_values('pct_strong', ascending=True)
            ec1, ec2 = st.columns(2)
            for col_e, df_e, label, clr in [(ec1, pos, "Emosi Positif", COLOR_GREEN), (ec2, neg, "Emosi Negatif", COLOR_RED)]:
                with col_e:
                    with st.container(border=True):
                        # FIX: label "Emosi Positif/Negatif" dengan warna kontras
                        st.markdown(
                            f"<div style='font-size:0.72rem;font-weight:700;color:{clr};"
                            f"margin-bottom:3px;padding:0'>{label}</div>",
                            unsafe_allow_html=True
                        )
                        if len(df_e) > 0:
                            pct_vals = df_e['pct_strong'] * 100 if df_e['pct_strong'].max() <= 1 else df_e['pct_strong']
                            fig_e = go.Figure(go.Bar(
                                x=pct_vals,
                                y=df_e['emosi'],
                                orientation='h',
                                marker_color=clr,
                                marker_opacity=0.85,
                                text=(pct_vals).round(1).astype(str) + '%',
                                textposition='outside',
                                # FIX: teks % di luar bar – gelap agar terbaca
                                textfont=dict(size=9, color="#111827", family="sans-serif"),
                                hovertemplate='<b>%{y}</b><br>%{text}<extra></extra>',
                            ))
                            fig_e = plotly_layout(fig_e, height=200, margin=dict(l=4, r=55, t=10, b=4))
                            fig_e.update_xaxes(
                                range=[0, pct_vals.max() * 1.18],
                                tickfont=dict(size=8, color="#374151"),
                            )
                            fig_e.update_yaxes(tickfont=dict(size=9, color="#111827"))
                            st.plotly_chart(fig_e, use_container_width=True, config={"displayModeBar":False})

    with em_tabs[1]:
        if len(emotion_seg) > 0:
            pivot_es = emotion_seg.pivot_table(
                index='segmen', columns='emosi', values='pct_strong', aggfunc='mean'
            ).fillna(0)
            if pivot_es.max().max() <= 1:
                pivot_es = pivot_es * 100
            fig_es = px.imshow(
                pivot_es,
                color_continuous_scale=[[0,'#fee2e2'],[0.5,'#fef9c3'],[1,'#dcfce7']],
                text_auto='.1f',
                aspect='auto',
            )
            # FIX: teks di dalam heatmap – gelap agar kontras di semua sel
            fig_es.update_traces(textfont=dict(size=9, color="#111827"))
            fig_es = plotly_layout(fig_es, height=220, margin=dict(l=4,r=4,t=18,b=4))
            fig_es.update_coloraxes(showscale=False)
            fig_es.update_xaxes(tickfont=dict(size=8.5, color="#374151"))
            fig_es.update_yaxes(tickfont=dict(size=9,   color="#111827"))
            st.plotly_chart(fig_es, use_container_width=True, config={"displayModeBar":False})

    with em_tabs[2]:
        if len(emotion_pan) > 0:
            pivot_ep = emotion_pan.pivot_table(
                index='panel', columns='emosi', values='pct_strong', aggfunc='mean'
            ).fillna(0)
            if pivot_ep.max().max() <= 1:
                pivot_ep = pivot_ep * 100
            fig_ep2 = px.imshow(
                pivot_ep,
                color_continuous_scale=[[0,'#fee2e2'],[0.5,'#fef9c3'],[1,'#dcfce7']],
                text_auto='.1f',
                aspect='auto',
            )
            # FIX: teks di dalam heatmap – gelap agar kontras
            fig_ep2.update_traces(textfont=dict(size=9, color="#111827"))
            fig_ep2 = plotly_layout(fig_ep2, height=180, margin=dict(l=4,r=4,t=18,b=4))
            fig_ep2.update_coloraxes(showscale=False)
            fig_ep2.update_xaxes(tickfont=dict(size=8.5, color="#374151"))
            fig_ep2.update_yaxes(tickfont=dict(size=9,   color="#111827"))
            st.plotly_chart(fig_ep2, use_container_width=True, config={"displayModeBar":False})

# ── AI ASSISTANT PANEL ────────────────────────────────────────────────────────
with ai_col:
    with st.container(border=True):
        # FIX: judul "AI Assistant" dan subtitle teks kontras – tidak putih di background terang
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
          <div style="width:26px;height:26px;
                      background:linear-gradient(135deg,{PRIMARY},{PRIMARY_LIGHT});
                      border-radius:7px;display:flex;align-items:center;
                      justify-content:center;flex-shrink:0;">
            {SVG_USERS.replace('stroke="currentColor"','stroke="white"')}
          </div>
          <div class="ai-panel-title">AI Assistant</div>
        </div>
        <div class="ai-panel-sub">Tanya seputar customer segments.</div>
        <div class="ai-panel-section-label">SUGGESTED QUESTIONS</div>
        """, unsafe_allow_html=True)

        if 'ci_chat' not in st.session_state:
            st.session_state.ci_chat = []

        for q in [
            "Segmen mana yang paling berisiko churn?",
            "Faktor apa yang paling mempengaruhi loyalitas?",
            "Segmen mana dengan potensi pertumbuhan terbesar?",
        ]:
            if st.button(q, key=f"ci_sq_{q}", use_container_width=True):
                risk_seg = segmen_prof.nsmallest(1, 'nps_score')['segmen'].values[0] if len(segmen_prof) > 0 else "N/A"
                best_seg = segmen_prof.nlargest(1, 'n')['segmen'].values[0]          if len(segmen_prof) > 0 else "N/A"
                ans = {
                    "Segmen mana yang paling berisiko churn?":
                        f"Paling berisiko: **{risk_seg}**. Churn risk saat ini: {churn_pct:.1f}%.",
                    "Faktor apa yang paling mempengaruhi loyalitas?":
                        f"Faktor utama: {', '.join(driver.head(3)['touchpoint'].tolist()) if len(driver)>0 else 'N/A'}.",
                    "Segmen mana dengan potensi pertumbuhan terbesar?":
                        f"Potensi terbesar: **{best_seg}** berdasarkan volume responden.",
                }
                st.session_state.ci_chat += [
                    {"role": "user", "content": q},
                    {"role": "ai",   "content": ans.get(q, "")},
                ]
                st.rerun()

        if st.session_state.ci_chat:
            chat_html = '<div class="chat-box">'
            for msg in st.session_state.ci_chat[-6:]:
                role_cls = "user" if msg["role"] == "user" else "ai"
                chat_html += f'<div class="chat-{role_cls}">{msg["content"]}</div>'
            st.markdown(chat_html + '</div>', unsafe_allow_html=True)

        user_q = st.text_input(
            "Ask...", key="ci_uq",
            label_visibility="collapsed",
            placeholder="Ask a question...",
        )
        if user_q:
            top_driver = driver.iloc[0]['touchpoint'] if len(driver) > 0 else 'N/A'
            st.session_state.ci_chat += [
                {"role": "user", "content": user_q},
                {"role": "ai",   "content":
                    f"Terdapat {total_seg} segmen aktif. "
                    f"Churn risk: {churn_pct:.1f}%. "
                    f"Driver loyalitas utama: {top_driver}."},
            ]
            st.rerun()
