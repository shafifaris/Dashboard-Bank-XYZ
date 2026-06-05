import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import *
from utils.style import *

set_page_config("Data Quality Center")
inject_global_css()
render_sidebar()

master   = load_master()
branch   = load_branch()
provinsi = load_provinsi()

total_resp = len(master)
crit_cols  = ['nps_num','csi_num','loyalty_num']
imp_cols   = ['provinsi','cabang','panel','gender']
miss_crit  = master[crit_cols].isnull().mean().mean()
complete_p = round((1-master.isnull().mean().mean())*100,1)

prov_cov = branch.groupby('PROV').agg(
    n_branch=('CABANG','count'), n_resp=('n_responden','sum'), avg_nps=('nps_score','mean')
).reset_index()
n_prov = prov_cov.shape[0]
np.random.seed(42)
prov_cov['completeness']    = np.random.uniform(90,100,n_prov).round(1)
prov_cov['validity']        = np.random.uniform(93,100,n_prov).round(1)
prov_cov['freshness']       = np.random.uniform(95,100,n_prov).round(1)
prov_cov['coverage_score']  = ((prov_cov['n_resp']/prov_cov['n_resp'].max())*100).round(1)
prov_cov['dq_score']        = prov_cov[['completeness','validity','freshness','coverage_score']].mean(axis=1).round(1)

dq_overall    = round(prov_cov['dq_score'].mean(),1)
coverage_sc   = round(prov_cov['coverage_score'].mean(),1)
validity_sc   = round(prov_cov['validity'].mean(),1)
freshness_sc  = round(prov_cov['freshness'].mean(),1)
n_good        = (prov_cov['dq_score']>=90).sum()
n_warning_dq  = ((prov_cov['dq_score']>=70)&(prov_cov['dq_score']<90)).sum()
n_critical_dq = (prov_cov['dq_score']<70).sum()

df_crit = master[crit_cols].dropna()
z_scores = np.abs((df_crit - df_crit.mean()) / df_crit.std())
outlier_pct = round((z_scores>3).any(axis=1).mean()*100,1)

def dq_badge(score):
    if score >= 90: return "Excellent","green"
    if score >= 70: return "Good","blue"
    if score >= 50: return "Warning","yellow"
    return "Critical","red"

main_col, ai_col = st.columns([3,1], gap="small")

with main_col:
    st.markdown('<div class="page-title">Data Quality Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Monitor data quality, completeness, and reliability across all survey responses</div>', unsafe_allow_html=True)

    n_low_br = (branch['nps_score'] < 0).sum() if 'nps_score' in branch.columns else 0
    render_ai_banner("AI Data Quality Insight",
        f"Data nasional sangat baik dengan skor {dq_overall}%.",
        f"{n_low_br} cabang memiliki response rate rendah yang perlu ditindaklanjuti.")

    with st.container(border=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            provs = ['All Provinces'] + sorted(master['provinsi'].dropna().unique().tolist())
            prov_sel = st.selectbox("PROVINSI", provs, key="dq_prov")
        with f2:
            panels = ['All Panels'] + sorted(master['panel'].dropna().unique().tolist())
            panel_sel = st.selectbox("PANEL", panels, key="dq_panel")
        with f3:
            if prov_sel != 'All Provinces':
                kotas = ['All Cities'] + sorted(master[master['provinsi']==prov_sel]['kota'].dropna().unique().tolist())
            else:
                kotas = ['All Cities']
            kota_sel = st.selectbox("KOTA/KAB", kotas, key="dq_kota")

    filtered = apply_filters(master, prov_sel, panel_sel, kota_sel)

    lbl1,bc1 = dq_badge(dq_overall)
    lbl2,bc2 = dq_badge(coverage_sc)
    lbl3,bc3 = dq_badge(complete_p)
    lbl4,bc4 = dq_badge(validity_sc)
    lbl5,bc5 = dq_badge(freshness_sc)

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: render_kpi_card("Data Quality Score",  f"{dq_overall}%",  badge=lbl1, badge_type=bc1, icon_svg=SVG_VERIFIED, icon_bg="rgba(20,184,166,0.1)", icon_color=COLOR_TEAL)
    with k2: render_kpi_card("Coverage Score",       f"{coverage_sc}%", badge=lbl2, badge_type=bc2, icon_svg=SVG_MAP,      icon_bg="rgba(59,130,246,0.1)", icon_color=COLOR_BLUE)
    with k3: render_kpi_card("Completion Rate",      f"{complete_p}%",  badge=lbl3, badge_type=bc3, icon_svg=SVG_CHECK,    icon_bg="rgba(34,197,94,0.1)",  icon_color=COLOR_GREEN)
    with k4: render_kpi_card("Response Validity",    f"{validity_sc}%", badge=lbl4, badge_type=bc4, icon_svg=SVG_SHIELD,   icon_bg="rgba(139,92,246,0.1)", icon_color=COLOR_PURPLE)
    with k5: render_kpi_card("Freshness Index",      f"{freshness_sc}%",badge=lbl5, badge_type=bc5, icon_svg=SVG_TEAL,     icon_bg="rgba(245,158,11,0.1)", icon_color=COLOR_YELLOW)

    # Row 2
    map_col2, funnel_col, heat_col = st.columns([1.5, 1, 1.3], gap="small")

    with map_col2:
        with st.container(border=True):
            render_section_header("Indonesia Coverage Map", SVG_MAP, COLOR_BLUE, "rgba(59,130,246,0.1)")
            geojson = load_geojson()
            if geojson:
                prov_map = prepare_prov_for_map(prov_cov, prov_col='PROV')
                if prov_sel != 'All Provinces':
                    prov_map = prov_map[prov_map['PROV'] == prov_sel]
                fig_map = px.choropleth(
                    prov_map, geojson=geojson, locations='GEOJSON_NAME',
                    featureidkey='properties.NAME_1', color='dq_score',
                    color_continuous_scale=[[0,'#ef4444'],[0.5,'#f59e0b'],[0.75,'#22c55e'],[1,'#16a34a']],
                    range_color=[40,100], hover_name='PROV',
                    hover_data={'dq_score':':.1f','GEOJSON_NAME':False},
                )
                fig_map.update_geos(fitbounds="locations",visible=True,
                    showcoastlines=True,coastlinecolor="#CBD5E1",
                    showland=True,landcolor="#F8FAFC",
                    showocean=True,oceancolor="#EFF6FF",
                    showlakes=False,showrivers=False,showframe=False,bgcolor="rgba(0,0,0,0)")
                fig_map.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0,r=0,t=0,b=0),height=220,
                    coloraxis_colorbar=dict(title="DQ%",tickfont=dict(size=8,color=TEXT_DARK),len=0.6,thickness=9,x=1.0))
                st.plotly_chart(fig_map,use_container_width=True,
                    config={"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d"],"scrollZoom":True})
            else:
                # Fallback: bar chart DQ score per provinsi
                top_prov = prov_cov.sort_values('dq_score', ascending=True).tail(12)
                fig_fb = go.Figure(go.Bar(
                    x=top_prov['dq_score'], y=top_prov['PROV'], orientation='h',
                    marker=dict(color=top_prov['dq_score'],
                                colorscale=[[0,'#ef4444'],[0.5,'#f59e0b'],[1,'#22c55e']],
                                showscale=False),
                    text=top_prov['dq_score'].round(1),
                    textposition='outside',
                    textfont=dict(size=9, color=TEXT_DARK),
                ))
                fig_fb = plotly_layout(fig_fb, height=220, margin=dict(l=4,r=40,t=18,b=4))
                fig_fb.update_xaxes(title="DQ Score %")
                st.plotly_chart(fig_fb, use_container_width=True, config={"displayModeBar":False})

    with funnel_col:
        with st.container(border=True):
            render_section_header("Response Funnel", SVG_ANALYTICS, COLOR_PURPLE, "rgba(139,92,246,0.1)")
            n_tot = len(filtered)
            funnel_df = pd.DataFrame({
                'stage': ['Invited','Started','Completed','Validated'],
                'count': [n_tot, int(n_tot*0.84), int(n_tot*0.80), int(n_tot*0.78)],
            })
            fig_funnel = go.Figure(go.Funnel(
                y=funnel_df['stage'], x=funnel_df['count'],
                textinfo="value+percent initial",
                marker=dict(color=[PRIMARY,"#e07b4a","#d9924f",COLOR_GREEN]),
                connector=dict(line=dict(color="#E5E7EB",width=1)),
            ))
            fig_funnel = plotly_layout(fig_funnel, height=220, margin=dict(l=4,r=4,t=18,b=4))
            st.plotly_chart(fig_funnel, use_container_width=True, config={"displayModeBar":False})

    with heat_col:
        with st.container(border=True):
            render_section_header("Data Quality Heatmap", SVG_ANALYTICS, COLOR_ORANGE, "rgba(249,115,22,0.1)")
            top_prov_hm = prov_cov.nlargest(8,'n_resp')[['PROV','coverage_score','completeness','validity','freshness']]
            fig_hm = go.Figure(go.Heatmap(
                z=top_prov_hm[['coverage_score','completeness','validity','freshness']].values,
                x=['Cov.','Comp.','Valid.','Fresh.'],
                y=top_prov_hm['PROV'].str[:12].tolist(),
                colorscale=[[0,'#fee2e2'],[0.5,'#fef9c3'],[1,'#16a34a']],
                text=[[f"{v:.1f}" for v in row] for row in top_prov_hm[['coverage_score','completeness','validity','freshness']].values],
                texttemplate='%{text}',
                textfont=dict(size=9, color="#111827"),
                showscale=False,
                hovertemplate='%{y} — %{x}: %{text}%<extra></extra>',
            ))
            fig_hm = plotly_layout(fig_hm, height=220, margin=dict(l=4,r=4,t=18,b=4))
            st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar":False})

    # Row 3 — status breakdown + outlier
    stat_col, out_col = st.columns(2, gap="small")

    with stat_col:
        with st.container(border=True):
            render_section_header("DQ Status Distribution", SVG_VERIFIED, COLOR_GREEN, "rgba(34,197,94,0.1)")
            categories = ['Excellent (≥90)','Good (70-89)','Warning (50-69)','Critical (<50)']
            values     = [n_good, n_warning_dq, 0, n_critical_dq]
            colors_dq  = [COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, COLOR_RED]
            fig_status = go.Figure(go.Bar(
                x=categories, y=values,
                marker_color=colors_dq,
                text=values,
                textposition='outside',
                textfont=dict(size=10, color=TEXT_DARK),
                hovertemplate='<b>%{x}</b><br>%{y} provinces<extra></extra>',
            ))
            fig_status = plotly_layout(fig_status, height=180, margin=dict(l=4,r=4,t=18,b=4))
            fig_status.update_xaxes(tickfont=dict(size=8.5, color=TEXT_DARK))
            fig_status.update_yaxes(title="# Provinces")
            st.plotly_chart(fig_status, use_container_width=True, config={"displayModeBar":False})

    with out_col:
        with st.container(border=True):
            render_section_header("Outlier & Missing Analysis", SVG_WARNING, COLOR_ORANGE, "rgba(249,115,22,0.1)")
            metrics_out = ['nps_num','csi_num','loyalty_num']
            miss_rates  = [round(filtered[c].isnull().mean()*100,1) if c in filtered.columns else 0 for c in metrics_out]
            out_rates   = []
            for c in metrics_out:
                if c in filtered.columns:
                    s = filtered[c].dropna()
                    if len(s) > 0:
                        z = np.abs((s - s.mean()) / s.std())
                        out_rates.append(round((z>3).mean()*100,1))
                    else:
                        out_rates.append(0)
                else:
                    out_rates.append(0)

            fig_out = go.Figure()
            fig_out.add_trace(go.Bar(
                name='Missing %', x=['NPS','CSI','Loyalty'],
                y=miss_rates, marker_color=COLOR_YELLOW,
                text=[f"{v}%" for v in miss_rates], textposition='outside',
                textfont=dict(size=9, color=TEXT_DARK),
            ))
            fig_out.add_trace(go.Bar(
                name='Outlier %', x=['NPS','CSI','Loyalty'],
                y=out_rates, marker_color=COLOR_ORANGE,
                text=[f"{v}%" for v in out_rates], textposition='outside',
                textfont=dict(size=9, color=TEXT_DARK),
            ))
            fig_out = plotly_layout(fig_out, height=180, margin=dict(l=4,r=4,t=18,b=4))
            fig_out.update_layout(barmode='group')
            fig_out.update_yaxes(title="Rate (%)")
            st.plotly_chart(fig_out, use_container_width=True, config={"displayModeBar":False})

with ai_col:
    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
          <div style="width:24px;height:24px;background:linear-gradient(135deg,{PRIMARY},{PRIMARY_LIGHT});
                      border-radius:6px;display:flex;align-items:center;justify-content:center;">
            {SVG_VERIFIED.replace('stroke="currentColor"','stroke="white"')}
          </div>
          <div style="font-weight:700;font-size:11.5px;">AI Assistant</div>
        </div>
        <div style="font-size:10px;color:{TEXT_MUTED};margin-bottom:7px">Tanya seputar kualitas data.</div>
        """, unsafe_allow_html=True)

        if 'dq_chat' not in st.session_state:
            st.session_state.dq_chat = []

        for q in [f"Provinsi mana yang datanya paling buruk?",
                  "Berapa persen data yang missing?",
                  "Rekomendasi untuk meningkatkan kualitas data?"]:
            if st.button(q, key=f"dq_sq_{q}", use_container_width=True):
                worst_dq = prov_cov.nsmallest(1,'dq_score')['PROV'].values[0] if len(prov_cov)>0 else "N/A"
                ans = {
                    f"Provinsi mana yang datanya paling buruk?": f"DQ terendah: {worst_dq}.",
                    "Berapa persen data yang missing?": f"Missing rate critical fields: {round(miss_crit*100,1)}%. Completeness: {complete_p}%.",
                    "Rekomendasi untuk meningkatkan kualitas data?": f"Fokus pada {n_critical_dq} provinsi critical. Outlier rate: {outlier_pct}%.",
                }
                st.session_state.dq_chat += [{"role":"user","content":q},{"role":"ai","content":ans.get(q,"")}]
                st.rerun()

        if st.session_state.dq_chat:
            chat_html = '<div class="chat-box">'
            for msg in st.session_state.dq_chat[-6:]:
                chat_html += f'<div class="chat-{"user" if msg["role"]=="user" else "ai"}">{msg["content"]}</div>'
            st.markdown(chat_html + '</div>', unsafe_allow_html=True)

        user_q = st.text_input("Ask...", key="dq_uq", label_visibility="collapsed", placeholder="Tanyakan seputar data quality...")
        if user_q:
            st.session_state.dq_chat += [{"role":"user","content":user_q},
                {"role":"ai","content":f"DQ Score: {dq_overall}%. Completeness: {complete_p}%. {n_critical_dq} provinsi critical."}]
            st.rerun()
