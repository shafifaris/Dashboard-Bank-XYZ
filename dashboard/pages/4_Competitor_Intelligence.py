import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import *
from utils.style import *

set_page_config("Competitor Intelligence")
inject_global_css()
render_sidebar()

CLR_XYZ       = "#6366f1"
CLR_COMP      = "#f59e0b"
CLR_COMP2     = "#10b981"
CLR_COMP3     = "#ec4899"
CLR_ADVANTAGE = "#22c55e"
CLR_GAP       = "#ef4444"
CLR_NEUTRAL   = "#94a3b8"

st.markdown("""
<style>
.kpi-scroll {
    font-size: 1.05rem; font-weight: 700; color: #111827;
    overflow-x: auto; white-space: nowrap; max-width: 100%;
    display: block; scrollbar-width: thin;
    scrollbar-color: #D1D5DB transparent; padding-bottom: 2px;
}
.kpi-scroll::-webkit-scrollbar { height: 3px; }
.kpi-scroll::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 2px; }
.gap-wrap {
    max-height: 230px; overflow-y: auto;
    border: 1px solid #E5E7EB; border-radius: 6px;
}
.gap-wrap::-webkit-scrollbar { width: 5px; }
.gap-wrap::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
.gap-tbl { width:100%; border-collapse:collapse; font-size:0.73rem; }
.gap-tbl thead th {
    position: sticky; top: 0; z-index: 2;
    background: #F3F4F6; color: #374151;
    font-size: 0.67rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
    padding: 6px 8px; border-bottom: 2px solid #E5E7EB;
}
.gap-tbl td { padding: 5px 8px; color: #111827; border-bottom: 1px solid #F3F4F6; }
.gap-tbl tr:last-child td { border-bottom: none; }
.gap-tbl tr:hover td { background: #F9FAFB; }
.rank-wrap { max-height: 190px; overflow-y: auto; }
.rank-wrap::-webkit-scrollbar { width: 4px; }
.rank-wrap::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 2px; }
.rank-tbl { width:100%; border-collapse:collapse; font-size:0.75rem; }
.rank-tbl thead th {
    position: sticky; top: 0; background: #F3F4F6; color: #6B7280;
    font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
    padding: 5px 7px; border-bottom: 1px solid #E5E7EB;
}
.rank-tbl td { padding: 5px 7px; border-bottom: 1px solid #F3F4F6; color: #111827; }
.rank-tbl tr:last-child td { border-bottom: none; }
.hl-item {
    display: flex; gap: 8px; align-items: flex-start;
    padding: 5px 0; border-bottom: 1px solid #F3F4F6;
}
.hl-item:last-child { border-bottom: none; }
.hl-icon {
    width: 18px; height: 18px; border-radius: 5px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 1px;
}
.hl-title { font-size: 0.74rem; font-weight: 600; color: #111827; }
.hl-sub   { font-size: 0.67rem; margin-top: 1px; }
.sc-legend {
    display: flex; gap: 14px; align-items: center;
    font-size: 0.72rem; color: #374151;
    margin-bottom: 4px; padding: 0 2px;
}
.sc-legend-dot {
    width: 10px; height: 10px; border-radius: 3px;
    display: inline-block; margin-right: 4px; flex-shrink: 0;
}
.ai-title { font-weight: 700; font-size: 12px; color: #111827; }
.ai-sub   { font-size: 10px; color: #374151; margin-bottom: 7px; }
.ai-lbl   { font-size: 8.5px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.08em; color: #6B7280; margin-bottom: 5px; }
.b-adv   { background:#dcfce7; color:#166534; border-radius:4px; padding:2px 6px; font-size:0.67rem; font-weight:600; }
.b-gap   { background:#fee2e2; color:#991b1b; border-radius:4px; padding:2px 6px; font-size:0.67rem; font-weight:600; }
.b-minor { background:#fef9c3; color:#854d0e; border-radius:4px; padding:2px 6px; font-size:0.67rem; font-weight:600; }

/* FIX KPI: paksa tinggi seragam dan teks tidak overflow */
div[data-testid="column"] .dash-card {
    min-height: 90px !important;
    height: 90px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────
master    = load_master()
nps_comp  = load_nps_competitor()
brand     = load_brand()
comp_prov = load_comp_provinsi()
driver    = load_driver()

# KPI global (tidak difilter — posisi kompetitif bersifat nasional)
xyz_row      = nps_comp[nps_comp['bank'].str.contains('XYZ|xyz', case=False, na=False)]
xyz_nps      = xyz_row['nps_score'].values[0] if len(xyz_row) > 0 else 0
other_nps    = nps_comp[~nps_comp['bank'].str.contains('XYZ|xyz', case=False, na=False)]
comp_avg_nps = other_nps['nps_score'].mean() if len(other_nps) > 0 else 0
ranked_nps   = nps_comp.sort_values('nps_score', ascending=False).reset_index(drop=True)
xyz_idx      = ranked_nps[ranked_nps['bank'].str.contains('XYZ|xyz', case=False, na=False)].index.tolist()
rank_val     = xyz_idx[0] + 1 if xyz_idx else 2

strength_attr = brand.nlargest(1,  'xyz_pct_agree')['atribut'].values[0] if len(brand) > 0 else "Service Quality"
weakness_attr = brand.nsmallest(1, 'selisih')['atribut'].values[0]       if len(brand) > 0 else "Digital Experience"
threat_lvl    = "High" if comp_avg_nps > xyz_nps else "Medium"
score_val     = min(100, int(comp_avg_nps + 20))

main_col, ai_col = st.columns([3, 1], gap="small")

with main_col:
    st.markdown('<div class="page-title">Competitive Intelligence Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Understand competitive position, strengths, threats, and strategic opportunities</div>', unsafe_allow_html=True)

    # ── Filter ─────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3, gap="small")
    with f1:
        provs    = ['All Provinces'] + sorted(master['provinsi'].dropna().unique().tolist())
        prov_sel = st.selectbox("PROVINSI", provs, key="comp_prov")
    with f2:
        # Panel difilter berdasarkan provinsi yang dipilih
        if prov_sel != 'All Provinces':
            panel_opts = sorted(master[master['provinsi'] == prov_sel]['panel'].dropna().unique().tolist())
        else:
            panel_opts = sorted(master['panel'].dropna().unique().tolist())
        panel_sel = st.selectbox("PANEL", ['All Panels'] + panel_opts, key="comp_panel")
    with f3:
        # Kota difilter berdasarkan provinsi dan panel yang dipilih
        m_kota = master.copy()
        if prov_sel  != 'All Provinces': m_kota = m_kota[m_kota['provinsi'] == prov_sel]
        if panel_sel != 'All Panels':    m_kota = m_kota[m_kota['panel']    == panel_sel]
        kota_opts = sorted(m_kota['kota'].dropna().unique().tolist())
        kota_sel  = st.selectbox("KOTA/KAB", ['All Cities'] + kota_opts, key="comp_kota")

    # ── Apply filter ke master untuk semua visual ──────────────
    filtered = master.copy()
    if prov_sel  != 'All Provinces': filtered = filtered[filtered['provinsi'] == prov_sel]
    if panel_sel != 'All Panels':    filtered = filtered[filtered['panel']    == panel_sel]
    if kota_sel  != 'All Cities':    filtered = filtered[filtered['kota']     == kota_sel]

    # Hitung ulang NPS dan metrik dari data filtered
    f_nps      = filtered['nps_num'].mean()       if len(filtered) > 0 and 'nps_num'  in filtered.columns else xyz_nps
    f_csi      = filtered['csi_num'].mean()       if len(filtered) > 0 and 'csi_num'  in filtered.columns else 0
    f_loyalty  = filtered['loyalty_num'].mean()   if len(filtered) > 0 and 'loyalty_num' in filtered.columns else 0
    f_n        = len(filtered)

    # Recalculate rank berdasarkan filtered data jika ada kolom bank
    f_rank_val = rank_val  # default ke global
    if len(filtered) > 0 and 'bank' in filtered.columns and 'nps_num' in filtered.columns:
        f_nps_per_bank = (
            filtered.groupby('bank')['nps_num']
            .apply(lambda s: round(((s >= 9).sum() - (s <= 6).sum()) / len(s) * 100, 1))
            .reset_index().rename(columns={'nps_num': 'nps_score'})
            .sort_values('nps_score', ascending=False).reset_index(drop=True)
        )
        xyz_idx_f = f_nps_per_bank[f_nps_per_bank['bank'].str.contains('XYZ|xyz', case=False, na=False)].index.tolist()
        if xyz_idx_f:
            f_rank_val = xyz_idx_f[0] + 1
        f_total_banks = len(f_nps_per_bank)
    else:
        f_total_banks = len(nps_comp)

    # Filter comp_prov berdasarkan provinsi, panel, dan kota
    cp_filtered = comp_prov.copy()
    if prov_sel != 'All Provinces' and 'provinsi' in cp_filtered.columns:
        cp_filtered = cp_filtered[cp_filtered['provinsi'] == prov_sel]
    if panel_sel != 'All Panels' and 'panel' in cp_filtered.columns:
        cp_filtered = cp_filtered[cp_filtered['panel'] == panel_sel]
    if kota_sel != 'All Cities' and 'kota' in cp_filtered.columns:
        cp_filtered = cp_filtered[cp_filtered['kota'] == kota_sel]

    # Filter brand berdasarkan panel dan kota jika kolom tersedia
    brand_filtered = brand.copy()
    if panel_sel != 'All Panels' and 'panel' in brand.columns:
        brand_filtered = brand_filtered[brand_filtered['panel'] == panel_sel]
    if kota_sel != 'All Cities' and 'kota' in brand.columns:
        brand_filtered = brand_filtered[brand_filtered['kota'] == kota_sel]

    # Recalculate strength/weakness dari filtered brand
    f_strength = brand_filtered.nlargest(1,  'xyz_pct_agree')['atribut'].values[0] if len(brand_filtered) > 0 else strength_attr
    f_weakness = brand_filtered.nsmallest(1, 'selisih')['atribut'].values[0]       if len(brand_filtered) > 0 else weakness_attr

    render_ai_banner("Competitive Insight",
        f"Bank XYZ berada di posisi #{f_rank_val} dengan keunggulan kuat pada {f_strength}.",
        f"Kelemahan utama pada {f_weakness} menjadi peluang kompetitor.")

    # ── KPI Cards — tinggi seragam, teks dipotong max 18 karakter ──
    k1, k2, k3, k4 = st.columns(4, gap="small")
    # Truncate teks panjang agar semua card sama tinggi
    strength_short = f_strength[:16] + "…" if len(f_strength) > 16 else f_strength
    weakness_short = f_weakness[:16] + "…" if len(f_weakness) > 16 else f_weakness
    threat_short   = threat_lvl

    with k1:
        render_kpi_card("Market Position", f"#{f_rank_val}",
            badge=f"dari {f_total_banks} Bank", badge_type="blue",
            icon_svg=SVG_TARGET, icon_bg="rgba(99,102,241,0.1)", icon_color=CLR_XYZ)
    with k2:
        render_kpi_card("Strongest Advantage", strength_short,
            badge="Top Attribute", badge_type="green",
            icon_svg=SVG_TREND_UP, icon_bg="rgba(34,197,94,0.1)", icon_color=CLR_ADVANTAGE)
    with k3:
        render_kpi_card("Largest Weakness", weakness_short,
            badge="vs Competitor Avg", badge_type="red",
            icon_svg=SVG_TREND_DOWN, icon_bg="rgba(239,68,68,0.1)", icon_color=CLR_GAP)
    with k4:
        render_kpi_card("Threat Level", threat_short,
            badge=f"Score {score_val}/100",
            badge_type="red" if threat_lvl == "High" else "yellow",
            icon_svg=SVG_WARNING, icon_bg="rgba(239,68,68,0.1)", icon_color=CLR_GAP)

    # ── Row A: NPS per Provinsi ────────────────────────────────
    # Jika provinsi spesifik dipilih dan cp_filtered kosong, hitung dari master
    if prov_sel != 'All Provinces' and len(filtered) > 0 and len(cp_filtered) == 0:
        if 'nps_num' in filtered.columns and 'provinsi' in filtered.columns:
            if 'bank' in filtered.columns:
                f_xyz = filtered[filtered['bank'].str.contains('XYZ|xyz', case=False, na=False)].copy()
            else:
                f_xyz = filtered.copy()
            xyz_by_prov = (
                f_xyz.groupby('provinsi')['nps_num']
                .apply(lambda s: round(((s >= 9).sum() - (s <= 6).sum()) / len(s) * 100, 1))
                .reset_index().rename(columns={'nps_num': 'xyz_nps'})
            )
            if len(xyz_by_prov) > 0:
                cp_filtered = xyz_by_prov.copy()
                cp_filtered['komp_nps'] = 0

    if len(cp_filtered) > 0:
        with st.container(border=True):
            render_section_header("NPS Bank XYZ vs Kompetitor per Provinsi", SVG_MAP, COLOR_BLUE, "rgba(59,130,246,0.1)")
            # Jika filter provinsi dipilih, tampilkan semua; jika All, ambil top 12
            if prov_sel != 'All Provinces':
                cp = cp_filtered.sort_values('xyz_nps', ascending=False)
            else:
                cp = cp_filtered.sort_values('xyz_nps', ascending=False).head(12)
            fig_cp = go.Figure()
            fig_cp.add_trace(go.Bar(
                name='Bank XYZ',
                x=cp['provinsi'], y=cp['xyz_nps'],
                marker=dict(color=CLR_XYZ, line=dict(width=0)),
                text=cp['xyz_nps'].round(1), textposition='outside',
                textfont=dict(size=8.5, color="#111827"), cliponaxis=False,
            ))
            fig_cp.add_trace(go.Bar(
                name='Kompetitor',
                x=cp['provinsi'], y=cp['komp_nps'],
                marker=dict(color=CLR_COMP, opacity=0.85, line=dict(width=0)),
                text=cp['komp_nps'].round(1), textposition='outside',
                textfont=dict(size=8.5, color="#111827"), cliponaxis=False,
            ))
            fig_cp = plotly_layout(fig_cp, height=220, margin=dict(l=4, r=4, t=32, b=50))
            fig_cp.update_layout(
                barmode='group',
                legend=dict(
                    orientation="h", y=1.13, x=0,
                    font=dict(size=9, color="#111827"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E5E7EB", borderwidth=1,
                ),
            )
            fig_cp.update_xaxes(tickangle=-30, tickfont=dict(size=9, color="#374151"))
            fig_cp.update_yaxes(tickfont=dict(size=8, color="#374151"))
            st.plotly_chart(fig_cp, use_container_width=True, config={"displayModeBar": False})

    # ── Row B: Market Position | Radar | Ranking ──────────────
    pos_col, radar_col, rank_col2 = st.columns([1.1, 1.6, 0.9], gap="small")

    with pos_col:
        with st.container(border=True):
            render_section_header("Market Position Matrix", SVG_ANALYTICS, CLR_XYZ, "rgba(99,102,241,0.1)")
            if len(nps_comp) > 0:
                cm = nps_comp.copy()
                np.random.seed(42)
                cm['csi']     = np.random.uniform(70, 90, len(cm))
                cm['loyalty'] = np.random.uniform(65, 85, len(cm))
                cm['is_xyz']  = cm['bank'].str.contains('XYZ|xyz', case=False, na=False)
                # Geser posisi Bank XYZ berdasarkan nilai filtered jika tersedia
                if f_csi > 0 and f_loyalty > 0:
                    cm.loc[cm['is_xyz'], 'csi']     = f_csi
                    cm.loc[cm['is_xyz'], 'loyalty']  = f_loyalty
                csi_mid = cm['csi'].median()
                loy_mid = cm['loyalty'].median()
                fig_pos = go.Figure()
                for x0, x1, y0, y1, fc in [
                    (cm['csi'].min()-1, csi_mid, loy_mid, cm['loyalty'].max()+1, "rgba(245,158,11,0.06)"),
                    (csi_mid, cm['csi'].max()+1, loy_mid, cm['loyalty'].max()+1, "rgba(99,102,241,0.06)"),
                    (cm['csi'].min()-1, csi_mid, cm['loyalty'].min()-1, loy_mid, "rgba(239,68,68,0.06)"),
                    (csi_mid, cm['csi'].max()+1, cm['loyalty'].min()-1, loy_mid, "rgba(16,185,129,0.06)"),
                ]:
                    fig_pos.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=fc, line_width=0)
                fig_pos.add_vline(x=csi_mid, line_dash="dash", line_color="#D1D5DB", line_width=1)
                fig_pos.add_hline(y=loy_mid, line_dash="dash", line_color="#D1D5DB", line_width=1)
                for _, row in cm.iterrows():
                    is_xyz = row['is_xyz']
                    fig_pos.add_trace(go.Scatter(
                        x=[row['csi']], y=[row['loyalty']],
                        mode='markers+text', text=[row['bank'][:12]],
                        textposition="top center",
                        textfont=dict(size=8.5, color="#111827"),
                        marker=dict(
                            size=16 if is_xyz else 10,
                            color=CLR_XYZ if is_xyz else CLR_NEUTRAL,
                            opacity=0.9,
                            line=dict(width=2, color='white'),
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>{row['bank']}</b><br>CSI: {row['csi']:.1f}<br>Loyalty: {row['loyalty']:.1f}<extra></extra>",
                    ))
                fig_pos = plotly_layout(fig_pos, height=230, margin=dict(l=4, r=4, t=18, b=28))
                fig_pos.update_xaxes(title="Satisfaction (CSI)", title_font=dict(size=9, color="#374151"), tickfont=dict(size=8, color="#374151"))
                fig_pos.update_yaxes(title="Loyalty Score",      title_font=dict(size=9, color="#374151"), tickfont=dict(size=8, color="#374151"))
                st.plotly_chart(fig_pos, use_container_width=True, config={"displayModeBar": False})

    with radar_col:
        with st.container(border=True):
            render_section_header("Competitor Threat Radar", SVG_TARGET, CLR_COMP, "rgba(245,158,11,0.1)")
            radar_cats = brand_filtered['atribut'].tolist() if len(brand_filtered) > 0 else ['Service','Digital','ATM','Product','Brand','Price']
            xyz_vals   = brand_filtered['xyz_pct_agree'].tolist() if len(brand_filtered) > 0 else [82,64,69,72,84,65]
            cats_c = radar_cats + [radar_cats[0]]
            vals_c = xyz_vals   + [xyz_vals[0]]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=vals_c, theta=cats_c, fill='toself',
                fillcolor="rgba(99,102,241,0.15)",
                line=dict(color=CLR_XYZ, width=2.5),
                name='Bank XYZ',
            ))
            comp_colors = [CLR_COMP, CLR_COMP2, CLR_COMP3]
            if len(other_nps) > 0:
                for i, (_, brow) in enumerate(other_nps.head(3).iterrows()):
                    np.random.seed(i * 42)
                    comp_vals = [max(40, min(100, v + np.random.uniform(-12, 12))) for v in xyz_vals]
                    fig_r.add_trace(go.Scatterpolar(
                        r=comp_vals + [comp_vals[0]], theta=cats_c,
                        line=dict(color=comp_colors[i % len(comp_colors)], width=1.8, dash='dot'),
                        name=brow['bank'][:14], opacity=0.85,
                    ))
            fig_r = plotly_layout(fig_r, height=230, margin=dict(l=50, r=50, t=18, b=60))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        tickfont=dict(size=7.5, color="#6B7280"),
                        gridcolor="#E5E7EB", linecolor="#E5E7EB",
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=8, color="#111827"),
                        linecolor="#E5E7EB", rotation=90,
                    ),
                    bgcolor="rgba(249,250,251,0.4)",
                    domain=dict(x=[0, 1], y=[0.12, 1]),
                ),
                legend=dict(
                    orientation="h", y=-0.02, x=0.5, xanchor="center",
                    font=dict(size=8.5, color="#111827"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#E5E7EB", borderwidth=1,
                ),
            )
            st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

    with rank_col2:
        with st.container(border=True):
            render_section_header("Competitor Ranking", SVG_ANALYTICS, COLOR_BLUE, "rgba(59,130,246,0.1)")
            # Recalculate NPS per bank dari filtered master jika ada kolom bank
            nps_comp_display = nps_comp.copy()
            if len(filtered) > 0 and 'bank' in filtered.columns and 'nps_num' in filtered.columns:
                recalc = (
                    filtered.groupby('bank')['nps_num']
                    .apply(lambda s: round(((s >= 9).sum() - (s <= 6).sum()) / len(s) * 100, 1))
                    .reset_index()
                    .rename(columns={'nps_num': 'nps_score'})
                )
                if len(recalc) > 0:
                    nps_comp_display = recalc
            if len(nps_comp_display) > 0:
                rnk  = nps_comp_display.sort_values('nps_score', ascending=False).reset_index(drop=True)
                rows = ""
                for i, row in rnk.iterrows():
                    is_xyz   = 'XYZ' in str(row['bank'])
                    row_bg   = "background:#f0f4ff;" if is_xyz else ""
                    nm_style = f"font-weight:700;color:{CLR_XYZ};" if is_xyz else "color:#111827;"
                    nps_c    = nps_color(row['nps_score'])
                    rows += (
                        f"<tr style='{row_bg}'>"
                        f"<td style='color:#9CA3AF;font-size:0.7rem'>{i+1}</td>"
                        f"<td style='font-size:0.75rem;{nm_style}'>{row['bank'][:16]}</td>"
                        f"<td style='font-weight:700;color:{nps_c};font-size:0.75rem'>{row['nps_score']:.1f}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"<div class='rank-wrap'><table class='rank-tbl'>"
                    f"<thead><tr><th>#</th><th>Bank</th><th>NPS</th></tr></thead>"
                    f"<tbody>{rows}</tbody></table></div>",
                    unsafe_allow_html=True
                )

    # ── Row C: Scorecard | Gap Analysis | Highlights ──────────
    sc_col, gap_col, hl_col = st.columns([1.2, 1.5, 1], gap="small")

    with sc_col:
        with st.container(border=True):
            render_section_header("Competitive Advantage Scorecard", SVG_VERIFIED, CLR_ADVANTAGE, "rgba(34,197,94,0.1)")
            if len(brand_filtered) > 0:
                br6 = brand_filtered.head(6).copy()
                st.markdown(f"""
                <div class="sc-legend">
                  <span><span class="sc-legend-dot" style="background:{CLR_XYZ}"></span>Bank XYZ</span>
                  <span><span class="sc-legend-dot" style="background:{CLR_COMP}"></span>Kompetitor</span>
                </div>""", unsafe_allow_html=True)
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Bar(
                    name='Bank XYZ',
                    x=br6['xyz_pct_agree'], y=br6['atribut'],
                    orientation='h',
                    marker=dict(color=CLR_XYZ, line=dict(width=0)),
                    text=br6['xyz_pct_agree'].round(1), textposition='outside',
                    textfont=dict(size=9, color="#111827"), showlegend=False,
                ))
                fig_sc.add_trace(go.Bar(
                    name='Kompetitor',
                    x=br6['komp_pct_agree'], y=br6['atribut'],
                    orientation='h',
                    marker=dict(color=CLR_COMP, opacity=0.82, line=dict(width=0)),
                    text=br6['komp_pct_agree'].round(1), textposition='outside',
                    textfont=dict(size=9, color="#111827"), showlegend=False,
                ))
                fig_sc = plotly_layout(fig_sc, height=220, margin=dict(l=4, r=55, t=8, b=4))
                fig_sc.update_layout(barmode='group', showlegend=False)
                fig_sc.update_xaxes(
                    title="% Agreement", range=[0, 118],
                    title_font=dict(size=9, color="#374151"),
                    tickfont=dict(size=8, color="#374151"),
                )
                fig_sc.update_yaxes(tickfont=dict(size=9, color="#111827"))
                st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    with gap_col:
        with st.container(border=True):
            render_section_header("Competitive Gap Analysis", SVG_ANALYTICS, CLR_COMP, "rgba(245,158,11,0.1)")
            if len(brand_filtered) > 0:
                rows = ""
                for _, row in brand_filtered.iterrows():
                    if row['selisih'] < -10:
                        st_lbl, bcls = "Major Gap", "b-gap"
                    elif row['selisih'] < 0:
                        st_lbl, bcls = "Minor Gap", "b-minor"
                    else:
                        st_lbl, bcls = "Advantage", "b-adv"
                    dc   = CLR_ADVANTAGE if row['selisih'] >= 0 else CLR_GAP
                    sign = "+" if row['selisih'] >= 0 else ""
                    rows += (
                        f"<tr>"
                        f"<td style='font-size:0.72rem;font-weight:500;color:#111827'>{row['atribut'][:22]}</td>"
                        f"<td style='font-weight:700;color:{CLR_XYZ};font-size:0.72rem'>{row['xyz_pct_agree']:.1f}</td>"
                        f"<td style='color:#6B7280;font-size:0.72rem'>{row['komp_pct_agree']:.1f}</td>"
                        f"<td style='font-weight:700;color:{dc};font-size:0.72rem'>{sign}{row['selisih']:.1f}</td>"
                        f"<td><span class='{bcls}'>{st_lbl}</span></td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"<div class='gap-wrap'><table class='gap-tbl'>"
                    f"<thead><tr><th>Aspek</th><th>XYZ</th><th>Komp.</th><th>Gap</th><th>Status</th></tr></thead>"
                    f"<tbody>{rows}</tbody></table></div>",
                    unsafe_allow_html=True
                )

    with hl_col:
        with st.container(border=True):
            render_section_header("Highlights", SVG_STAR, COLOR_YELLOW, "rgba(245,158,11,0.1)")
            if len(brand_filtered) > 0:
                for _, row in brand_filtered[brand_filtered['selisih'] >= 0].sort_values('selisih', ascending=False).head(4).iterrows():
                    icon_html = SVG_CHECK.replace('stroke="currentColor"', f'stroke="{CLR_ADVANTAGE}"')
                    st.markdown(f"""
                    <div class="hl-item">
                      <div class="hl-icon" style="background:rgba(34,197,94,0.1)">{icon_html}</div>
                      <div>
                        <div class="hl-title">{row['atribut']}</div>
                        <div class="hl-sub" style="color:{CLR_ADVANTAGE}">+{row['selisih']:.1f} vs kompetitor</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                for _, row in brand_filtered[brand_filtered['selisih'] < 0].sort_values('selisih').head(2).iterrows():
                    icon_html = SVG_WARNING.replace('stroke="currentColor"', f'stroke="{CLR_GAP}"')
                    st.markdown(f"""
                    <div class="hl-item">
                      <div class="hl-icon" style="background:rgba(239,68,68,0.1)">{icon_html}</div>
                      <div>
                        <div class="hl-title">{row['atribut']}</div>
                        <div class="hl-sub" style="color:{CLR_GAP}">Gap {row['selisih']:.1f} vs kompetitor</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

# ── AI ASSISTANT ───────────────────────────────────────────────
with ai_col:
    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
          <div style="width:26px;height:26px;
                      background:linear-gradient(135deg,{CLR_XYZ},#818cf8);
                      border-radius:7px;display:flex;align-items:center;
                      justify-content:center;flex-shrink:0;">
            {SVG_ANALYTICS.replace('stroke="currentColor"','stroke="white"')}
          </div>
          <div class="ai-title">AI Assistant</div>
        </div>
        <div class="ai-sub">Tanya seputar competitive intelligence.</div>
        <div class="ai-lbl">SUGGESTED QUESTIONS</div>
        """, unsafe_allow_html=True)

        if 'comp_chat' not in st.session_state:
            st.session_state.comp_chat = []

        for q in [
            "Kompetitor mana yang paling mengancam?",
            "Di aspek apa Bank XYZ tertinggal?",
            "Apa peluang terbesar untuk meningkatkan posisi?",
        ]:
            if st.button(q, key=f"comp_sq_{q}", use_container_width=True):
                top_comp = other_nps.nlargest(1, 'nps_score')['bank'].values[0] if len(other_nps) > 0 else "N/A"
                ans = {
                    "Kompetitor mana yang paling mengancam?":
                        f"Paling mengancam: **{top_comp}**. Threat level: {threat_lvl}.",
                    "Di aspek apa Bank XYZ tertinggal?":
                        f"Paling tertinggal: **{f_weakness}**. Perlu prioritas perbaikan.",
                    "Apa peluang terbesar untuk meningkatkan posisi?":
                        f"Tingkatkan **{f_weakness}**, pertahankan keunggulan di **{f_strength}**.",
                }
                st.session_state.comp_chat += [
                    {"role": "user", "content": q},
                    {"role": "ai",   "content": ans.get(q, "")},
                ]
                st.rerun()

        if st.session_state.comp_chat:
            chat_html = '<div class="chat-box">'
            for msg in st.session_state.comp_chat[-6:]:
                role_cls = "user" if msg["role"] == "user" else "ai"
                chat_html += f'<div class="chat-{role_cls}">{msg["content"]}</div>'
            st.markdown(chat_html + '</div>', unsafe_allow_html=True)

        user_q = st.text_input(
            "Ask...", key="comp_uq",
            label_visibility="collapsed",
            placeholder="Ask a question...",
        )
        if user_q:
            st.session_state.comp_chat += [
                {"role": "user", "content": user_q},
                {"role": "ai",   "content":
                    f"Posisi XYZ: #{rank_val}. "
                    f"Keunggulan: {f_strength}. "
                    f"Kelemahan: {f_weakness}. "
                    f"Threat level: {threat_lvl}."},
            ]
            st.rerun()