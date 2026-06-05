import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import io, json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.data_loader import *
from utils.style import *

set_page_config("Report Center")
inject_global_css()
render_sidebar()

master   = load_master()
branch   = load_branch()
provinsi = load_provinsi()
driver   = load_driver()
brand    = load_brand()
nps_comp = load_nps_competitor()
ipa      = load_ipa()

nps_val = nps_score(master['nps_num']) or 0.0
csi_val = round(master['csi_num'].mean(),1)
loy_val = round(master['loyalty_num'].mean(),1)
n_resp  = len(master)
n_branch = branch.shape[0]
n_prov   = master['provinsi'].nunique()
today    = datetime.now()
top_drv  = driver.iloc[0]['touchpoint'] if len(driver)>0 else "N/A"
worst_prov = provinsi.nsmallest(1,'nps_score')['PROV'].values[0] if len(provinsi)>0 else "N/A"

SCHEDULED = [
    {"name":"Daily Executive Summary",    "freq":"Daily",     "next":(today+timedelta(1)).strftime("%b %d 09:00"),   "rec":8,  "status":"Active"},
    {"name":"Weekly Performance Summary", "freq":"Weekly",    "next":(today+timedelta(7)).strftime("%b %d 09:00"),   "rec":15, "status":"Active"},
    {"name":"Monthly Executive Deck",     "freq":"Monthly",   "next":(today+timedelta(30)).strftime("%b %d 09:00"),  "rec":12, "status":"Active"},
    {"name":"Quarterly Business Review",  "freq":"Quarterly", "next":(today+timedelta(90)).strftime("%b %d 09:00"),  "rec":10, "status":"Active"},
]
RECENT = [
    {"name":"Executive Summary — May 2026",       "type":"PDF",  "size":"2.4 MB","date":"Today 09:02"},
    {"name":"Branch Performance — May 2026",      "type":"PPTX", "size":"5.7 MB","date":"Today 09:58"},
    {"name":"Customer Intelligence — May 2026",   "type":"PDF",  "size":"3.1 MB","date":"Yesterday 16:45"},
    {"name":"Competitor Analysis — May 2026",     "type":"PPTX", "size":"4.6 MB","date":"Yesterday 14:20"},
    {"name":"Touchpoint Performance — May 2026",  "type":"PDF",  "size":"2.7 MB","date":"May 27, 2026"},
]

def gen_exec_csv():
    buf = io.StringIO()
    buf.write(f"BankXYZ Executive Summary,{today.strftime('%B %d %Y')}\n\n")
    buf.write(f"NPS,{nps_val:.1f}\nCSI,{csi_val:.1f}\nLoyalty,{loy_val:.1f}\nResponses,{n_resp}\n\n")
    buf.write(provinsi.nlargest(10,'nps_score')[['PROV','nps_score']].to_csv(index=False))
    return buf.getvalue().encode()

def gen_excel():
    try:
        import openpyxl
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            pd.DataFrame({'Metric':['NPS','CSI','Loyalty','Responses'],
                          'Value':[nps_val,csi_val,loy_val,n_resp]}).to_excel(w,sheet_name='KPI',index=False)
            branch.to_excel(w, sheet_name='Branch', index=False)
            provinsi.to_excel(w, sheet_name='Province', index=False)
            driver.to_excel(w, sheet_name='Drivers', index=False)
        return buf.getvalue()
    except:
        return gen_exec_csv()

main_col, ai_col = st.columns([3,1], gap="small")

with main_col:
    st.markdown('<div class="page-title">Report Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Generate, customize, and distribute executive reports with AI-powered insights</div>', unsafe_allow_html=True)

    f1,f2,f3 = st.columns(3)
    with f1:
        provs = ['All Provinces'] + sorted(master['provinsi'].dropna().unique().tolist())
        prov_sel = st.selectbox("PROVINSI", provs, key="rc_prov")
    with f2:
        panels = ['All Panels'] + sorted(master['panel'].dropna().unique().tolist())
        panel_sel = st.selectbox("PANEL", panels, key="rc_panel")
    with f3:
        if prov_sel != 'All Provinces':
            kotas = ['All Cities'] + sorted(master[master['provinsi']==prov_sel]['kota'].dropna().unique().tolist())
        else:
            kotas = ['All Cities']
        kota_sel = st.selectbox("KOTA/KAB", kotas, key="rc_kota")

    render_ai_banner("AI Executive Summary",
        f"CSI nasional meningkat +3.2 poin menjadi {csi_val:.1f}.",
        f"Peningkatan didorong oleh {top_drv}. Potensi NPS: +6.4 poin.")

    # Generate buttons
    gb1, gb2 = st.columns(2)
    with gb1:
        st.download_button("Generate Board Report (CSV)", data=gen_exec_csv(),
            file_name=f"BankXYZ_Executive_{today.strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True)
    with gb2:
        excel_data = gen_excel()
        st.download_button("Export Full Data (Excel)", data=excel_data,
            file_name=f"BankXYZ_Full_{today.strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

    # KPI Stats
    ks1,ks2,ks3,ks4 = st.columns(4)
    for col_k,(lbl,val) in zip([ks1,ks2,ks3,ks4],[
        ("Reports Generated","24"),("Scheduled","12"),("Downloads","156"),("Recipients","28")
    ]):
        with col_k:
            render_kpi_card(lbl, val, badge="+vs Apr", badge_type="green",
                icon_svg=SVG_REPORT, icon_bg="rgba(59,130,246,0.1)", icon_color=COLOR_BLUE)

    # Report Builder + Schedule
    rb_col, sched_col = st.columns([1, 1.3], gap="small")

    with rb_col:
        with st.container(border=True):
            render_section_header("Report Builder", SVG_REPORT, COLOR_BLUE, "rgba(59,130,246,0.1)")
            RTYPES = [
                ("Executive Summary",      "executive_summary"),
                ("Branch Performance",     "branch_performance"),
                ("Customer Intelligence",  "customer_intelligence"),
                ("Competitor Analysis",    "competitor_analysis"),
                ("Touchpoint Performance", "touchpoint_performance"),
                ("Custom Report",          "custom"),
            ]
            sel = st.session_state.get('sel_report','executive_summary')
            for title_r, key_r in RTYPES:
                bg_r = f"background:#fff7ed;" if key_r==sel else ""
                if st.button(title_r, key=f"rb_{key_r}", use_container_width=True):
                    st.session_state['sel_report'] = key_r
                    st.rerun()
            sel = st.session_state.get('sel_report','executive_summary')
            st.download_button("Generate Report", data=gen_exec_csv(),
                file_name=f"BankXYZ_{sel}_{today.strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True, type="primary")

    with sched_col:
        with st.container(border=True):
            render_section_header("Schedule Manager", SVG_TEAL, COLOR_TEAL, "rgba(20,184,166,0.1)")
            rows = "".join([
                f"<tr><td style='font-size:10px;font-weight:500'>{s['name']}</td>"
                f"<td style='font-size:9.5px;color:{TEXT_MUTED}'>{s['freq']}</td>"
                f"<td style='font-size:9.5px;color:{TEXT_MUTED}'>{s['next']}</td>"
                f"<td style='text-align:center'>{s['rec']}</td>"
                f"<td><span class='badge badge-green'>{s['status']}</span></td></tr>"
                for s in SCHEDULED
            ])
            st.markdown(f"""<table class="styled-table">
              <thead><tr><th>Report</th><th>Freq</th><th>Next</th><th>Recip.</th><th>Status</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

    # Distribution + Recent + Export
    dist_col, recent_col, export_col = st.columns([1.2,1.3,1], gap="small")

    with dist_col:
        with st.container(border=True):
            render_section_header("Distribution Center", SVG_USERS, COLOR_PURPLE, "rgba(139,92,246,0.1)")
            for name_r, role_r, sent_r in [
                ("Budi Santoso","CEO","Today 09:02"),
                ("Dewi Lestari","COO","Today 08:02"),
                ("Andi Pratama","Regional Head","Today 08:03"),
                ("Rina Handayani","Region Mgr","Today 09:04"),
            ]:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:6px;padding:4px 0;border-bottom:1px solid #F9FAFB">
                  <div style="width:22px;height:22px;background:{PRIMARY};border-radius:50%;
                              display:flex;align-items:center;justify-content:center;flex-shrink:0">
                    <span style="color:white;font-size:0.65rem;font-weight:700">{name_r[0]}</span>
                  </div>
                  <div style="flex:1;min-width:0">
                    <div style="font-size:10px;font-weight:600">{name_r}</div>
                    <div style="font-size:9px;color:{TEXT_MUTED}">{role_r} · {sent_r}</div>
                  </div>
                  <span class="badge badge-green">Sent</span>
                </div>""", unsafe_allow_html=True)

    with recent_col:
        with st.container(border=True):
            render_section_header("Recent Reports", SVG_REPORT, COLOR_ORANGE, "rgba(249,115,22,0.1)")
            for rpt in RECENT:
                is_pdf = rpt['type']=='PDF'
                clr_r  = COLOR_RED if is_pdf else PRIMARY
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #F9FAFB">
                  <div style="display:flex;gap:6px;align-items:center">
                    <div style="width:24px;height:24px;background:{'#fee2e2' if is_pdf else '#fff7ed'};border-radius:5px;
                                display:flex;align-items:center;justify-content:center;flex-shrink:0">
                      <span style="font-size:0.6rem;font-weight:700;color:{clr_r}">{rpt['type']}</span>
                    </div>
                    <div>
                      <div style="font-size:10px;font-weight:600">{rpt['name']}</div>
                      <div style="font-size:9px;color:{TEXT_MUTED}">{rpt['date']} · {rpt['size']}</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            st.download_button("Download All (CSV)", data=gen_exec_csv(),
                file_name=f"BankXYZ_All_{today.strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True)

    with export_col:
        with st.container(border=True):
            render_section_header("Export Center", SVG_BOLT, PRIMARY, "rgba(200,65,11,0.1)")
            json_data = json.dumps({
                "report_date": today.strftime('%Y-%m-%d'),
                "kpi":{"nps":nps_val,"csi":csi_val,"loyalty":loy_val,"responses":n_resp},
                "top_drivers": driver.head(5)[['touchpoint','correlation']].to_dict('records') if len(driver)>0 else [],
            }, indent=2).encode()

            for fmt_r, desc_r, data_r, fname_r, mime_r in [
                ("CSV",   "Raw Data",          gen_exec_csv(),   f"BankXYZ_{today.strftime('%Y%m%d')}.csv",  "text/csv"),
                ("Excel", "Data & Pivot",      gen_excel(),      f"BankXYZ_{today.strftime('%Y%m%d')}.xlsx", "text/csv"),
                ("JSON",  "API-ready Format",  json_data,        f"BankXYZ_{today.strftime('%Y%m%d')}.json", "application/json"),
            ]:
                c1, c2 = st.columns([3,1])
                with c1:
                    st.markdown(f"""
                    <div style="padding:5px 0;border-bottom:1px solid #F9FAFB">
                      <div style="font-weight:700;font-size:10.5px;color:{PRIMARY}">{fmt_r}</div>
                      <div style="font-size:9px;color:{TEXT_MUTED}">{desc_r}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.download_button("↓", data=data_r, file_name=fname_r,
                        mime=mime_r, key=f"exp_{fmt_r}")

    # Activity Timeline
    with st.container(border=True):
        render_section_header("Report Activity Timeline", SVG_TEAL, COLOR_TEAL, "rgba(20,184,166,0.1)")
        for time_r, clr_r, title_r, desc_r in [
            ("09:00",COLOR_GREEN,"Executive Summary Generated","oleh AI Copilot"),
            ("09:02",PRIMARY,"Report Sent to 8 Recipients","via Email"),
            ("09:05",COLOR_BLUE,"Branch Report Downloaded","oleh Andi Pratama"),
            ("10:00",COLOR_YELLOW,"Monthly Deck Generated","12 Slides"),
            ("10:02",COLOR_GREEN,"Schedule Updated","Next: Jun 2, 2026"),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #F9FAFB">
              <div style="width:38px;text-align:right;font-size:9px;color:{TEXT_MUTED};flex-shrink:0;padding-top:2px">{time_r}</div>
              <div style="width:7px;height:7px;background:{clr_r};border-radius:50%;margin-top:4px;flex-shrink:0"></div>
              <div>
                <div style="font-size:10.5px;font-weight:600">{title_r}</div>
                <div style="font-size:9px;color:{TEXT_MUTED}">{desc_r}</div>
              </div>
            </div>""", unsafe_allow_html=True)

with ai_col:
    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
          <div style="width:24px;height:24px;background:linear-gradient(135deg,{PRIMARY},{PRIMARY_LIGHT});
                      border-radius:6px;display:flex;align-items:center;justify-content:center;">
            {SVG_REPORT.replace('stroke="currentColor"','stroke="white"')}
          </div>
          <div style="font-weight:700;font-size:11.5px;">AI Copilot</div>
        </div>
        <div style="font-size:8.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{TEXT_MUTED};margin-bottom:5px">SUGGESTED ACTIONS</div>
        """, unsafe_allow_html=True)

        if 'rc_chat' not in st.session_state:
            st.session_state.rc_chat = []

        for q in ["Generate executive summary bulan ini",
                  "Buat laporan performa cabang",
                  "Insight apa untuk dilaporkan ke direksi?"]:
            if st.button(q, key=f"rc_sq_{q[:25]}", use_container_width=True):
                ans = {
                    "Generate executive summary bulan ini":
                        f"Summary siap. NPS {nps_val:.1f}, CSI {csi_val:.1f}, Loyalty {loy_val:.1f}. Klik 'Generate Board Report'.",
                    "Buat laporan performa cabang":
                        f"{n_branch} cabang. Klik 'Export Full Data' untuk data lengkap.",
                    "Insight apa untuk dilaporkan ke direksi?":
                        f"NPS {nps_val:.1f} (target 75.0). Perhatian: {worst_prov}. Driver utama: {top_drv}.",
                }
                st.session_state.rc_chat += [{"role":"user","content":q},{"role":"ai","content":ans.get(q,"")}]
                st.rerun()

        if st.session_state.rc_chat:
            chat_html = '<div class="chat-box">'
            for msg in st.session_state.rc_chat[-6:]:
                chat_html += f'<div class="chat-{"user" if msg["role"]=="user" else "ai"}">{msg["content"]}</div>'
            st.markdown(chat_html + '</div>', unsafe_allow_html=True)

        user_q = st.text_input("Tanya...", key="rc_uq", label_visibility="collapsed", placeholder="Tanyakan sesuatu...")
        if user_q:
            st.session_state.rc_chat += [{"role":"user","content":user_q},
                {"role":"ai","content":f"NPS {nps_val:.1f}, CSI {csi_val:.1f}. {len(RECENT)} recent reports tersedia."}]
            st.rerun()

