import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import *
from utils.style import *

set_page_config("Executive Action Center")
inject_global_css()
render_sidebar()

# ── Load data ────────────────────────────────────────────────────────────────
master = load_master()
driver = load_driver()
branch = load_branch()

# Optional data sources (graceful fallback)
try:    ipa_df   = load_ipa()
except: ipa_df   = None
try:    comp_df  = load_nps_competitor()
except: comp_df  = None
try:    emo_df   = load_emotion()
except: emo_df   = None
try:    dig_df   = load_digitalisasi()
except: dig_df   = None
try:    sw_df    = load_switching()
except: sw_df    = None

# ── Derived metrics ───────────────────────────────────────────────────────────
nps_val  = nps_score(master['nps_num']) or 0.0
churn_r  = round((master['nps_num'] <= 6).sum() / len(master) * 100, 1)

# Branch health summary
try:
    branch['_status'] = branch.apply(lambda r: get_branch_status(r['nps_score']), axis=1)
    n_critical = (branch['_status'] == 'Critical').sum()
    n_warning  = (branch['_status'] == 'Warning').sum()
    n_healthy  = (branch['_status'] == 'Healthy').sum()
except:
    n_critical, n_warning, n_healthy = 0, 0, len(branch)

# Segmen Detractor
try:
    detractors = master[master['nps_num'] <= 6]
    seg_col = [c for c in master.columns if 'segmen' in c.lower() or 'segment' in c.lower()]
    if seg_col:
        seg_detract = detractors.groupby(seg_col[0]).size().sort_values(ascending=False)
        top_detract_seg  = seg_detract.index[0] if len(seg_detract) > 0 else "N/A"
        top_detract_n    = int(seg_detract.iloc[0]) if len(seg_detract) > 0 else 0
    else:
        top_detract_seg, top_detract_n = "N/A", 0
except:
    top_detract_seg, top_detract_n = "N/A", 0

# IPA quick view — High Importance, Low Performance
try:
    imp_col  = [c for c in ipa_df.columns if 'import' in c.lower() or 'importance' in c.lower()][0]
    perf_col = [c for c in ipa_df.columns if 'perfor' in c.lower() or 'satisf' in c.lower()][0]
    tp_col   = [c for c in ipa_df.columns if 'touch' in c.lower() or 'item' in c.lower()][0]
    imp_med  = ipa_df[imp_col].median()
    perf_med = ipa_df[perf_col].median()
    ipa_critical = ipa_df[(ipa_df[imp_col] >= imp_med) & (ipa_df[perf_col] < perf_med)]
    ipa_critical_list = ipa_critical[tp_col].tolist()[:3]
except:
    ipa_critical_list = []

# Competitor NPS gap
try:
    nps_col_comp = [c for c in comp_df.columns if 'nps' in c.lower()][0]
    name_col     = [c for c in comp_df.columns if 'bank' in c.lower() or 'name' in c.lower()][0]
    comp_avg     = comp_df[comp_df[name_col].str.upper().str.contains('XYZ|BANK', na=False) == False][nps_col_comp].mean()
    bankxyz_nps  = comp_df[comp_df[name_col].str.upper().str.contains('XYZ', na=False)][nps_col_comp].mean()
    comp_gap     = round(bankxyz_nps - comp_avg, 1)
except:
    comp_gap = None

# Emotion alert
try:
    emo_col = [c for c in emo_df.columns if 'emosi' in c.lower() or 'emotion' in c.lower() or 'sentimen' in c.lower()][0]
    neg_emotions = emo_df[emo_df[emo_col].str.lower().str.contains('negatif|negative|kecewa|frustr|marah', na=False)]
    dom_neg_emotion = neg_emotions[emo_col].value_counts().index[0] if len(neg_emotions) > 0 else None
except:
    dom_neg_emotion = None

# Digital adoption
try:
    dig_col = [c for c in dig_df.columns if 'digital' in c.lower() or 'channel' in c.lower()][0]
    dig_pct = round((dig_df[dig_col].str.lower().str.contains('digital', na=False)).sum() / len(dig_df) * 100, 1)
except:
    dig_pct = None

# Switching risk
try:
    risk_col = [c for c in sw_df.columns if 'risk' in c.lower() or 'inten' in c.lower() or 'switch' in c.lower()][0]
    n_high_risk = (sw_df[risk_col].str.lower().str.contains('high|tinggi', na=False)).sum()
except:
    n_high_risk = None

# ── Priority actions ──────────────────────────────────────────────────────────
PRIORITY_ACTIONS = []
for i, (_, row) in enumerate(driver.head(5).iterrows()):
    PRIORITY_ACTIONS.append({
        "rank":       i+1,
        "action":     f"Improve {row['touchpoint']}",
        "desc":       f"Enhance {row['touchpoint'].lower()} quality and availability",
        "nps_impact": round(row['abs_corr']*15,1),
        "loy_impact": round(row['abs_corr']*8,1),
        "effort":     ["Medium","Medium","High","High","Low"][min(i,4)],
        "timeline":   [90,60,120,150,90][min(i,4)],
        "owner":      ["Digital Banking","Operations","Product","Finance","Marketing"][min(i,4)],
        "priority":   ["High Priority","High Priority","Medium Priority","Medium Priority","Low Priority"][min(i,4)],
        "status":     ["In Progress","In Progress","Not Started","In Progress","Completed"][min(i,4)],
    })

total_nps_impact = sum(a['nps_impact'] for a in PRIORITY_ACTIONS)
n_completed      = sum(1 for a in PRIORITY_ACTIONS if a['status']=='Completed')
n_in_progress    = sum(1 for a in PRIORITY_ACTIONS if a['status']=='In Progress')
n_not_started    = sum(1 for a in PRIORITY_ACTIONS if a['status']=='Not Started')
overall_progress = round((n_completed*100+n_in_progress*50)/(len(PRIORITY_ACTIONS)*100)*100)

# Derived insight values
bottleneck_pct   = round(n_completed / len(PRIORITY_ACTIONS) * 100)
longest_timeline = max(PRIORITY_ACTIONS, key=lambda a: a['timeline'])
best_roi_action  = max(PRIORITY_ACTIONS, key=lambda a: a['nps_impact'] / ({"Low":1,"Medium":2,"High":3}[a['effort']]))
high_effort_low_nps = [a for a in PRIORITY_ACTIONS if a['effort']=='High' and a['nps_impact'] < (total_nps_impact/len(PRIORITY_ACTIONS))]
revenue_potential = round(total_nps_impact * 0.22, 2)
churn_nps_gap     = round(total_nps_impact * 0.9 - churn_r / 2, 1)

# Owners appearing in multiple phases (dependency chain)
from collections import Counter
all_owners = [a['owner'] for a in PRIORITY_ACTIONS]
owner_counts = Counter(all_owners)
dep_chain_owners = [o for o, c in owner_counts.items() if c >= 2]

# ── Layout ────────────────────────────────────────────────────────────────────
main_col, ai_col = st.columns([3,1], gap="small")

with main_col:
    st.markdown('<div class="page-title">Executive Action Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Prioritize actions, estimate impact, and drive strategic execution</div>', unsafe_allow_html=True)

    # ── FIX 1: AI Insight Banner — improved contrast ──────────────────────────
    top_act  = PRIORITY_ACTIONS[0]['action'] if PRIORITY_ACTIONS else "N/A"
    top_act2 = PRIORITY_ACTIONS[1]['action'] if len(PRIORITY_ACTIONS)>1 else "N/A"

    # Custom high-contrast banner replacing render_ai_banner to fix readability
    st.markdown(f"""
    <div style="background:#FFF7ED;border:1px solid #FED7AA;border-left:4px solid {PRIMARY};
                border-radius:10px;padding:10px 14px;margin-bottom:10px;
                display:flex;align-items:flex-start;gap:10px;">
      <div style="width:28px;height:28px;background:linear-gradient(135deg,{PRIMARY},{PRIMARY_LIGHT});
                  border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
        </svg>
      </div>
      <div>
        <div style="font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:0.1em;
                    color:{PRIMARY};margin-bottom:2px;">⚡ EXECUTIVE INSIGHT</div>
        <div style="font-size:11px;font-weight:600;color:#1C1917;">
          Fokus utama: <strong>{top_act}</strong> dan <strong>{top_act2}</strong>.
        </div>
        <div style="font-size:10.5px;color:#57534E;margin-top:1px;">
          Proyeksi peningkatan NPS sebesar <strong style="color:{COLOR_GREEN}">+{total_nps_impact:.1f} poin</strong> jika semua aksi dieksekusi.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: render_kpi_card("Total Actions", str(len(PRIORITY_ACTIONS)), badge="+2 vs Apr", badge_type="blue", icon_svg=SVG_BOLT, icon_bg="rgba(200,65,11,0.1)", icon_color=PRIMARY)
    with k2: render_kpi_card("Est. NPS Impact", f"+{total_nps_impact:.1f}", badge="+2.8 vs Apr", badge_type="green", icon_svg=SVG_TREND_UP, icon_bg="rgba(34,197,94,0.1)", icon_color=COLOR_GREEN)
    with k3: render_kpi_card("Loyalty Growth", f"+{sum(a['loy_impact'] for a in PRIORITY_ACTIONS):.1f}%", badge="+1.6% vs Apr", badge_type="green", icon_svg=SVG_HEART, icon_bg="rgba(139,92,246,0.1)", icon_color=COLOR_PURPLE)
    with k4: render_kpi_card("Churn Reduction", f"-{churn_r/2:.1f}%", badge="-1.2% vs Apr", badge_type="green", icon_svg=SVG_SHIELD, icon_bg="rgba(34,197,94,0.1)", icon_color=COLOR_GREEN)
    with k5: render_kpi_card("Overall Progress", f"{overall_progress}%", badge="High", badge_type="green", icon_svg=SVG_VERIFIED, icon_bg="rgba(20,184,166,0.1)", icon_color=COLOR_TEAL)

    # Impact-Effort Matrix + Priority Table
    ie_col, act_col = st.columns([1, 2], gap="small")

    with ie_col:
        with st.container(border=True):
            render_section_header("Impact vs Effort Matrix", SVG_TARGET, COLOR_ORANGE, "rgba(249,115,22,0.1)")
            fig_ie = go.Figure()
            for q_shape in [
                (0,0.5,0.5,1.0,"rgba(34,197,94,0.06)"),
                (0.5,1.0,0.5,1.0,"rgba(245,158,11,0.06)"),
                (0,0.5,0,0.5,"rgba(107,114,128,0.04)"),
                (0.5,1.0,0,0.5,"rgba(239,68,68,0.05)"),
            ]:
                fig_ie.add_shape(type="rect",x0=q_shape[0],x1=q_shape[1],
                                  y0=q_shape[2],y1=q_shape[3],fillcolor=q_shape[4],line_width=0)
            fig_ie.add_vline(x=0.5,line_dash="dash",line_color="#E5E7EB",line_width=1)
            fig_ie.add_hline(y=0.5,line_dash="dash",line_color="#E5E7EB",line_width=1)
            effort_num = {"Low":0.2,"Medium":0.5,"High":0.8}
            prio_clr   = {"High Priority":COLOR_RED,"Medium Priority":COLOR_YELLOW,"Low Priority":COLOR_GREEN}
            for act in PRIORITY_ACTIONS:
                ef  = effort_num.get(act['effort'],0.5)
                imp = min(0.95, act['nps_impact']/(total_nps_impact+1))
                clr = prio_clr.get(act['priority'],TEXT_MUTED)
                fig_ie.add_trace(go.Scatter(
                    x=[ef],y=[imp],mode='markers+text',
                    text=[act['action'].replace('Improve ','')[:10]],
                    textposition="top center",
                    textfont=dict(size=7.5,color=TEXT_DARK),
                    marker=dict(size=11,color=clr,line=dict(width=1.5,color='white')),
                    showlegend=False,
                    hovertemplate=f"<b>{act['action']}</b><br>Impact: +{act['nps_impact']:.1f} NPS<br>Effort: {act['effort']}<extra></extra>",
                ))
            for ann in [("Quick Wins",0.25,0.95,COLOR_GREEN),("Strategic",0.75,0.95,COLOR_YELLOW),
                        ("Low Prio",0.25,0.05,TEXT_MUTED),("Major Proj",0.75,0.05,COLOR_ORANGE)]:
                fig_ie.add_annotation(x=ann[1],y=ann[2],text=ann[0],showarrow=False,
                    font=dict(size=8,color=ann[3]),opacity=0.7)
            fig_ie = plotly_layout(fig_ie,height=240,margin=dict(l=4,r=4,t=18,b=28),show_legend=False)
            fig_ie.update_xaxes(title="Effort",range=[0,1],tickvals=[0.2,0.5,0.8],ticktext=["Low","Med","High"])
            fig_ie.update_yaxes(title="Impact (NPS)",range=[0,1])
            st.plotly_chart(fig_ie,use_container_width=True,config={"displayModeBar":False})

    with act_col:
        with st.container(border=True):
            render_section_header("Top Priority Actions", SVG_BOLT, PRIMARY, "rgba(200,65,11,0.1)")
            status_badge = {"Completed":"green","In Progress":"blue","Not Started":"yellow","Delayed":"red"}
            rows = ""
            for act in PRIORITY_ACTIONS:
                sb    = status_badge.get(act['status'],"yellow")
                eff_b = "red" if act['effort']=="High" else ("yellow" if act['effort']=="Medium" else "green")
                rows += f"""<tr>
                  <td style="text-align:center;font-weight:700;color:{TEXT_MUTED}">{act['rank']}</td>
                  <td>
                    <div style="font-weight:600;font-size:10px">{act['action']}</div>
                    <div style="font-size:9px;color:{TEXT_MUTED}">{act['desc']}</div>
                  </td>
                  <td style="font-weight:700;color:{COLOR_GREEN};text-align:center">+{act['nps_impact']}</td>
                  <td><span class="badge badge-{eff_b}">{act['effort']}</span></td>
                  <td style="font-size:9.5px;color:{TEXT_MUTED};text-align:center">{act['timeline']}d</td>
                  <td style="font-size:9.5px;color:{TEXT_MUTED}">{act['owner'][:12]}</td>
                  <td><span class="badge badge-{sb}">{act['status']}</span></td>
                </tr>"""
            st.markdown(f"""
            <table class="styled-table">
              <thead><tr>
                <th>#</th><th>Action</th><th>NPS</th><th>Effort</th>
                <th>Days</th><th>Owner</th><th>Status</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

    # Projected Impact
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    render_section_header("Projected Business Impact", SVG_TREND_UP, COLOR_GREEN, "rgba(34,197,94,0.1)")
    pi_cols = st.columns(5)
    projections = [
        ("NPS Increase",       f"+{total_nps_impact:.1f}",   COLOR_GREEN),
        ("Loyalty Growth",     f"+{sum(a['loy_impact'] for a in PRIORITY_ACTIONS):.1f}%", COLOR_GREEN),
        ("Churn Reduction",    f"-{churn_r/2:.1f}%",          COLOR_RED),
        ("Retention",          f"+{round(total_nps_impact*0.9,1)}%", COLOR_GREEN),
        ("Revenue Potential",  f"+{revenue_potential:.2f}M", COLOR_GREEN),
    ]
    for col_p, (lbl, val, clr) in zip(pi_cols, projections):
        with col_p:
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORDER};border-radius:10px;
                        padding:10px;text-align:center">
              <div style="font-size:8.5px;font-weight:700;text-transform:uppercase;
                          color:{TEXT_MUTED};margin-bottom:3px">{lbl}</div>
              <div style="font-size:20px;font-weight:700;color:{clr}">{val}</div>
            </div>""", unsafe_allow_html=True)

    # ── FIX 2: Executive Roadmap — distinct phase colors & readable pills ─────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    render_section_header("Executive Roadmap", SVG_REPORT, COLOR_BLUE, "rgba(59,130,246,0.1)")
    r1, r2, r3 = st.columns(3, gap="small")
    roadmap_phases = [
        ("30 Days",     "Quick Wins",         COLOR_GREEN,  "#DCFCE7", "#166534", round(total_nps_impact*0.25,1), PRIORITY_ACTIONS[:3]),
        ("90 Days",     "Key Initiatives",    COLOR_YELLOW, "#FEF9C3", "#854D0E", round(total_nps_impact*0.45,1), PRIORITY_ACTIONS[1:4]),
        ("6-12 Months", "Strategic Projects", COLOR_BLUE,   "#DBEAFE", "#1E40AF", round(total_nps_impact*0.30,1), PRIORITY_ACTIONS[2:5]),
    ]
    for col_r, (period, lbl, border_clr, pill_bg, pill_text, est, items) in zip([r1,r2,r3], roadmap_phases):
        with col_r:
            pills_html = "".join([
                f"<span style='display:inline-block;background:{pill_bg};color:{pill_text};"
                f"border-radius:5px;padding:2px 7px;font-size:9.5px;font-weight:600;"
                f"margin:2px 2px 2px 0;border:1px solid {border_clr}20;'>"
                f"● {a['action']}</span>"
                for a in items
            ])
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORDER};border-radius:10px;
                        padding:11px;border-left:4px solid {border_clr};">
              <div style="font-weight:700;font-size:13px;color:{border_clr}">{period}</div>
              <div style="font-size:9px;font-weight:700;color:{TEXT_MUTED};text-transform:uppercase;
                          letter-spacing:0.07em;margin-bottom:6px">{lbl}</div>
              <div style="margin-bottom:6px">{pills_html}</div>
              <div style="margin-top:6px;display:flex;justify-content:space-between;
                          border-top:1px solid #F3F4F6;padding-top:5px">
                <span style="font-size:9px;color:{TEXT_MUTED}">Est. NPS Impact</span>
                <span style="font-weight:700;font-size:11px;color:{COLOR_GREEN}">+{est}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── NEW: 8 Strategic Insights ─────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    render_section_header("Strategic Insights", SVG_TARGET, COLOR_ORANGE, "rgba(249,115,22,0.1)")

    ins_cols = st.columns(4, gap="small")

    insight_cards = [
        {
            "icon": "⚠️",
            "title": "Bottleneck Eksekusi",
            "value": f"{n_completed}/{len(PRIORITY_ACTIONS)}",
            "value_color": COLOR_RED,
            "desc": f"Hanya {n_completed} aksi selesai. {n_in_progress} masih In Progress, {n_not_started} belum dimulai.",
            "border": COLOR_RED,
        },
        {
            "icon": "⏱️",
            "title": "Risiko Timeline",
            "value": f"{longest_timeline['timeline']} hari",
            "value_color": COLOR_ORANGE,
            "desc": f"{longest_timeline['action']} adalah aksi terlama. Dimulai sekarang berarti selesai jauh melewati target 90 hari.",
            "border": COLOR_ORANGE,
        },
        {
            "icon": "📈",
            "title": "NPS per Effort (ROI)",
            "value": f"+{best_roi_action['nps_impact']}",
            "value_color": COLOR_GREEN,
            "desc": f"{best_roi_action['action']} = ROI terbaik ({best_roi_action['effort']} effort, NPS tertinggi). Prioritas saat ini sudah tepat.",
            "border": COLOR_GREEN,
        },
        {
            "icon": "👥",
            "title": "Distribusi Owner",
            "value": f"{len(set(all_owners))} Tim",
            "value_color": COLOR_BLUE,
            "desc": f"{len(PRIORITY_ACTIONS)} aksi dikelola oleh {len(set(all_owners))} tim berbeda. Koordinasi lintas-tim jadi faktor kritis.",
            "border": COLOR_BLUE,
        },
    ]

    for col_i, card in zip(ins_cols, insight_cards):
        with col_i:
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORDER};border-radius:10px;padding:10px;
                        border-top:3px solid {card['border']};height:100%">
              <div style="font-size:9px;font-weight:700;text-transform:uppercase;color:{TEXT_MUTED};
                          margin-bottom:3px">{card['icon']} {card['title']}</div>
              <div style="font-size:18px;font-weight:800;color:{card['value_color']};
                          margin-bottom:4px">{card['value']}</div>
              <div style="font-size:9.5px;color:{TEXT_DARK};line-height:1.4">{card['desc']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    ins_cols2 = st.columns(4, gap="small")

    insight_cards2 = [
        {
            "icon": "🔴",
            "title": "High Effort ≠ High NPS",
            "value": f"{len(high_effort_low_nps)} Aksi",
            "value_color": COLOR_RED,
            "desc": (
                f"{', '.join([a['action'] for a in high_effort_low_nps[:2]])} — effort High tapi NPS di bawah rata-rata. Perlu evaluasi ulang."
                if high_effort_low_nps else "Semua aksi High Effort memiliki NPS yang proporsional."
            ),
            "border": COLOR_RED,
        },
        {
            "icon": "💰",
            "title": "Revenue Gap",
            "value": f"Rp {revenue_potential:.2f}M",
            "value_color": COLOR_GREEN,
            "desc": f"Potensi revenue Rp {revenue_potential:.2f}M hanya tercapai jika semua {len(PRIORITY_ACTIONS)} aksi selesai. Bottleneck saat ini memblok peluang ini.",
            "border": COLOR_GREEN,
        },
        {
            "icon": "🔗",
            "title": "Dependency Chain",
            "value": f"{len(dep_chain_owners)} Owner",
            "value_color": COLOR_ORANGE,
            "desc": (
                f"{', '.join(dep_chain_owners)} muncul di beberapa fase roadmap. Keterlambatan 1 berdampak ke semua fase."
                if dep_chain_owners else "Tidak ada ketergantungan owner lintas fase yang terdeteksi."
            ),
            "border": COLOR_ORANGE,
        },
        {
            "icon": "📉",
            "title": "Churn vs NPS Gap",
            "value": f"-{churn_r/2:.1f}%",
            "value_color": COLOR_RED,
            "desc": f"NPS proyeksi naik +{total_nps_impact:.1f} tapi churn hanya turun -{churn_r/2:.1f}%. Gap ini perlu diinvestigasi — ada faktor retensi yang belum tertangkap.",
            "border": "#F59E0B",
        },
    ]

    for col_i2, card in zip(ins_cols2, insight_cards2):
        with col_i2:
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORDER};border-radius:10px;padding:10px;
                        border-top:3px solid {card['border']};height:100%">
              <div style="font-size:9px;font-weight:700;text-transform:uppercase;color:{TEXT_MUTED};
                          margin-bottom:3px">{card['icon']} {card['title']}</div>
              <div style="font-size:18px;font-weight:800;color:{card['value_color']};
                          margin-bottom:4px">{card['value']}</div>
              <div style="font-size:9.5px;color:{TEXT_DARK};line-height:1.4">{card['desc']}</div>
            </div>""", unsafe_allow_html=True)

    # ── NEW: Data-Source Insights (Branch, IPA, Competitor, Emotion, Digital, Switching) ──
    has_extra = any([n_critical > 0, ipa_critical_list, comp_gap is not None,
                     dom_neg_emotion, dig_pct is not None, n_high_risk is not None,
                     top_detract_seg != "N/A"])

    if has_extra:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        render_section_header("Early Warning Indicators", SVG_SHIELD, COLOR_RED, "rgba(239,68,68,0.1)")
        extra_cols = st.columns(3, gap="small")
        extra_cards = []

        # Branch health
        if n_critical > 0 or n_warning > 0:
            total_br = n_critical + n_warning + n_healthy
            extra_cards.append({
                "icon": "🏢",
                "title": "Branch Health",
                "value": f"{n_critical} Kritis",
                "value_color": COLOR_RED if n_critical > 0 else COLOR_YELLOW,
                "desc": f"{n_critical} cabang Critical, {n_warning} Warning dari {total_br} total cabang. Butuh eskalasi segera.",
                "border": COLOR_RED if n_critical > 0 else COLOR_YELLOW,
            })

        # IPA critical
        if ipa_critical_list:
            extra_cards.append({
                "icon": "📊",
                "title": "IPA — High Imp, Low Perf",
                "value": f"{len(ipa_critical_list)} Item",
                "value_color": COLOR_ORANGE,
                "desc": f"Touchpoint kritis: {', '.join(ipa_critical_list)}. Penting bagi nasabah tapi performa rendah.",
                "border": COLOR_ORANGE,
            })

        # Competitor gap
        if comp_gap is not None:
            extra_cards.append({
                "icon": "🏆",
                "title": "Competitor NPS Gap",
                "value": f"{'+' if comp_gap >= 0 else ''}{comp_gap}",
                "value_color": COLOR_GREEN if comp_gap >= 0 else COLOR_RED,
                "desc": f"BankXYZ {'unggul' if comp_gap >= 0 else 'tertinggal'} {abs(comp_gap)} poin dari rata-rata kompetitor. {'Pertahankan keunggulan.' if comp_gap >= 0 else 'Perlu strategi akselerasi.'}",
                "border": COLOR_GREEN if comp_gap >= 0 else COLOR_RED,
            })

        # Segmen detractor
        if top_detract_seg != "N/A":
            extra_cards.append({
                "icon": "👤",
                "title": "Segmen Detractor",
                "value": str(top_detract_seg),
                "value_color": COLOR_RED,
                "desc": f"Segmen '{top_detract_seg}' punya {top_detract_n} detractor terbanyak (NPS ≤6). Target utama untuk program churn reduction.",
                "border": COLOR_RED,
            })

        # Emotion alert
        if dom_neg_emotion:
            extra_cards.append({
                "icon": "😠",
                "title": "Emotion Alert",
                "value": str(dom_neg_emotion),
                "value_color": COLOR_ORANGE,
                "desc": f"Emosi negatif dominan: '{dom_neg_emotion}'. Perlu response cepat dari tim Customer Experience.",
                "border": COLOR_ORANGE,
            })

        # Digital adoption
        if dig_pct is not None:
            extra_cards.append({
                "icon": "📱",
                "title": "Digital Adoption",
                "value": f"{dig_pct}%",
                "value_color": COLOR_BLUE if dig_pct >= 50 else COLOR_ORANGE,
                "desc": f"{dig_pct}% nasabah pakai channel digital. {'Mayoritas digital — penting untuk mendukung aksi Digital Banking.' if dig_pct >= 50 else 'Adopsi digital masih rendah. Aksi Digital Banking harus diprioritaskan.'}",
                "border": COLOR_BLUE if dig_pct >= 50 else COLOR_ORANGE,
            })

        # Switching risk
        if n_high_risk is not None:
            extra_cards.append({
                "icon": "⚡",
                "title": "Switching Risk",
                "value": f"{n_high_risk} Nasabah",
                "value_color": COLOR_RED,
                "desc": f"{n_high_risk} nasabah dengan intensi switching tinggi terdeteksi. Perlu program retensi segera sebelum churn terjadi.",
                "border": COLOR_RED,
            })

        # Render up to 6 extra cards in rows of 3
        for i in range(0, min(6, len(extra_cards)), 3):
            chunk = extra_cards[i:i+3]
            row_cols = st.columns(len(chunk), gap="small")
            for col_e, card in zip(row_cols, chunk):
                with col_e:
                    st.markdown(f"""
                    <div style="background:white;border:1px solid {BORDER};border-radius:10px;padding:10px;
                                border-top:3px solid {card['border']};height:100%">
                      <div style="font-size:9px;font-weight:700;text-transform:uppercase;color:{TEXT_MUTED};
                                  margin-bottom:3px">{card['icon']} {card['title']}</div>
                      <div style="font-size:18px;font-weight:800;color:{card['value_color']};
                                  margin-bottom:4px">{card['value']}</div>
                      <div style="font-size:9.5px;color:{TEXT_DARK};line-height:1.4">{card['desc']}</div>
                    </div>""", unsafe_allow_html=True)

    # Action Monitoring
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    render_section_header("Action Monitoring", SVG_VERIFIED, COLOR_TEAL, "rgba(20,184,166,0.1)")
    m_cols = st.columns(5)
    for col_m, (status, n_s, clr) in zip(m_cols,[
        ("Completed",   n_completed,   COLOR_GREEN),
        ("In Progress", n_in_progress, COLOR_BLUE),
        ("Not Started", n_not_started, TEXT_MUTED),
        ("Delayed",     0,             COLOR_RED),
        ("Overall",     overall_progress, PRIMARY),
    ]):
        with col_m:
            pct  = round(n_s/len(PRIORITY_ACTIONS)*100) if status!="Overall" else n_s
            disp = f"{n_s}" if status != "Overall" else f"{n_s}%"
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORDER};border-radius:10px;
                        padding:9px;text-align:center">
              <div style="font-size:8.5px;font-weight:700;text-transform:uppercase;color:{TEXT_MUTED};margin-bottom:2px">{status}</div>
              <div style="font-size:20px;font-weight:700;color:{clr}">{disp}</div>
              <div style="background:#F3F4F6;border-radius:4px;height:4px;margin-top:5px">
                <div style="background:{clr};width:{min(100,pct if status!='Overall' else n_s)}%;height:100%;border-radius:4px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

# ── AI Assistant Column ───────────────────────────────────────────────────────
with ai_col:
    with st.container(border=True):
        # FIX: High-contrast AI Assistant header
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
          <div style="width:24px;height:24px;background:linear-gradient(135deg,{PRIMARY},{PRIMARY_LIGHT});
                      border-radius:6px;display:flex;align-items:center;justify-content:center;">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
            </svg>
          </div>
          <div style="font-weight:700;font-size:11.5px;color:{TEXT_DARK};">AI Assistant</div>
        </div>
        <div style="font-size:8.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
                    color:{TEXT_DARK};background:#F3F4F6;border-radius:5px;padding:4px 7px;
                    margin-bottom:6px;">💬 SUGGESTED QUESTIONS</div>
        """, unsafe_allow_html=True)

        if 'act_chat' not in st.session_state:
            st.session_state.act_chat = []

        suggested_qs = [
            f"Mengapa {PRIORITY_ACTIONS[0]['action'].replace('Improve ','')} menjadi prioritas?",
            "Quick win apa yang bisa dijalankan 30 hari?",
            "Simulasikan dampak semua prioritas.",
            f"Apa risiko terbesar dari {longest_timeline['action']}?",
        ]

        for q in suggested_qs:
            if st.button(q, key=f"act_sq_{q[:28]}", use_container_width=True):
                ans_map = {
                    f"Mengapa {PRIORITY_ACTIONS[0]['action'].replace('Improve ','')} menjadi prioritas?":
                        f"Korelasi tertinggi r={driver.iloc[0]['correlation']:.2f}. Est. +{PRIORITY_ACTIONS[0]['nps_impact']:.1f} NPS.",
                    "Quick win apa yang bisa dijalankan 30 hari?":
                        f"Quick wins: {', '.join([a['action'] for a in PRIORITY_ACTIONS if a['effort']=='Low'][:2]) or PRIORITY_ACTIONS[0]['action']}. Fokus yang effortnya rendah tapi impactnya tinggi.",
                    "Simulasikan dampak semua prioritas.":
                        f"Jika semua dijalankan: NPS +{total_nps_impact:.1f}, revenue +Rp {revenue_potential:.2f}M, progress {overall_progress}%.",
                    f"Apa risiko terbesar dari {longest_timeline['action']}?":
                        f"{longest_timeline['action']} butuh {longest_timeline['timeline']} hari. Risiko: melebihi target 90 hari, berdampak ke fase Strategic. Pertimbangkan reschedule atau tambah resource.",
                }
                st.session_state.act_chat += [
                    {"role":"user","content":q},
                    {"role":"ai","content":ans_map.get(q, f"{len(PRIORITY_ACTIONS)} aksi, progress {overall_progress}%, NPS est. +{total_nps_impact:.1f}.")}
                ]
                st.rerun()

        if st.session_state.act_chat:
            chat_html = '<div class="chat-box">'
            for msg in st.session_state.act_chat[-6:]:
                chat_html += f'<div class="chat-{"user" if msg["role"]=="user" else "ai"}">{msg["content"]}</div>'
            st.markdown(chat_html + '</div>', unsafe_allow_html=True)

        user_q = st.text_input("Ask...", key="act_uq", label_visibility="collapsed", placeholder="Ask a question...")
        if user_q:
            st.session_state.act_chat += [
                {"role":"user","content":user_q},
                {"role":"ai","content":f"{len(PRIORITY_ACTIONS)} aksi, progress {overall_progress}%, NPS est. +{total_nps_impact:.1f}. Bottleneck: {n_completed}/{len(PRIORITY_ACTIONS)} selesai."}
            ]
            st.rerun()
