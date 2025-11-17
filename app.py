# -*- coding: utf-8 -*-
# app.py
# Corrected — Polished Compliance-Gated Mission Control (Streamlit prototype)
# Note: this file intentionally avoids stray replacement glyphs and emojis to ensure Streamlit Cloud accepts it.
 
import streamlit as st
import pandas as pd
import io
import json
import zipfile
import datetime
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from uuid import uuid4
from reportlab.lib.utils import ImageReader
 
# Page config (no emoji to avoid encoding edge-cases)
st.set_page_config(layout="wide", page_title="Compliance-Gated Mission Control")
 
# -------------------------
# Embedded sample dataset
# -------------------------
SAMPLE_CSV = """date,tail,sw_version,aid_count,fault_code,humidity,ata_chapter,flight_hours,oat,airport
2025-10-16,N101,v1.9,0,FC100,40,31,12.4,22,KATL
2025-10-17,N102,v2.0,1,FC200,60,31,8.2,25,KJFK
2025-10-18,N103,v1.9,0,FC100,35,31,11.0,20,KLAX
2025-10-19,N104,v2.0,0,FC300,55,32,9.5,18,KORD
2025-10-20,N105,v2.1,2,FC100,78,31,13.2,30,KSFO
2025-10-21,N106,v1.9,0,FC200,45,31,10.1,19,KMIA
2025-10-22,N107,v2.1,1,FC300,82,31,14.3,31,KSEA
2025-10-23,N108,v1.9,0,FC100,38,31,7.6,21,KBOS
2025-10-24,N101,v2.1,2,FC200,85,31,12.9,33,KATL
2025-10-25,N102,v2.1,3,FC100,80,31,9.9,32,KJFK
2025-10-26,N103,v1.9,0,FC300,30,32,11.5,16,KLAX
2025-10-27,N104,v2.0,1,FC200,65,31,8.8,24,KORD
2025-10-28,N105,v2.1,2,FC100,75,31,13.5,29,KSFO
2025-10-29,N106,v1.9,0,FC400,42,31,10.7,20,KMIA
2025-10-30,N107,v2.1,1,FC100,70,31,14.0,28,KSEA
2025-10-31,N108,v1.9,0,FC200,50,32,7.9,23,KBOS
2025-11-01,N101,v2.1,4,FC100,88,31,13.6,34,KATL
2025-11-02,N102,v2.1,2,FC200,82,31,9.7,31,KJFK
2025-11-03,N103,v1.9,0,FC300,43,31,11.2,19,KLAX
2025-11-04,N104,v2.0,1,FC100,60,31,9.1,22,KORD
2025-11-05,N105,v2.1,5,FC100,90,31,13.8,35,KSFO
2025-11-06,N106,v1.9,0,FC200,37,31,10.5,18,KMIA
2025-11-07,N107,v2.1,1,FC300,68,31,14.2,27,KSEA
2025-11-08,N108,v1.9,0,FC100,44,31,8.1,21,KBOS
2025-11-09,N101,v2.1,2,FC200,79,31,12.5,30,KATL
2025-11-10,N102,v2.1,1,FC100,83,31,9.6,32,KJFK
2025-11-11,N103,v1.9,0,FC400,39,32,11.0,17,KLAX
2025-11-12,N104,v2.0,0,FC200,58,31,9.3,24,KORD
2025-11-13,N105,v2.1,3,FC100,86,31,13.1,33,KSFO
2025-11-14,N106,v1.9,0,FC300,48,31,10.4,20,KMIA
"""
 
# -------------------------
# Data helpers
# -------------------------
@st.cache_data
def load_df():
    df = pd.read_csv(io.StringIO(SAMPLE_CSV), parse_dates=["date"])
    df['aid_count'] = df['aid_count'].astype(int)
    return df.sort_values("date")
 
def daily_series(df):
    s = df.groupby(df['date'].dt.date)['aid_count'].sum().reset_index()
    s.columns = ['date', 'aid_count']
    s['date'] = pd.to_datetime(s['date'])
    return s
 
def sw_before_after(df, cutoff_days=15):
    maxd = df['date'].max()
    cutoff = maxd - pd.Timedelta(days=cutoff_days)
    before = df[df['date'] < cutoff].groupby('sw_version')['aid_count'].sum()
    after = df[df['date'] >= cutoff].groupby('sw_version')['aid_count'].sum()
    versions = sorted(set(before.index).union(set(after.index)))
    rows = []
    for v in versions:
        rows.append({
            'sw_version': v,
            'before': int(before.get(v, 0)),
            'after': int(after.get(v, 0)),
            'delta': int(after.get(v, 0)) - int(before.get(v, 0))
        })
    return pd.DataFrame(rows), cutoff
 
def v21_stats(df, cutoff_days=15):
    maxd = df['date'].max()
    cutoff = maxd - pd.Timedelta(days=cutoff_days)
    recent = df[df['date'] >= cutoff]
    total = recent['aid_count'].sum() or 1
    v21 = recent[recent['sw_version'] == 'v2.1']['aid_count'].sum()
    pct = 100.0 * v21 / total
    return int(v21), int(total), float(pct)
 
def humidity_correlation(df, sw='v2.1', cutoff_days=15):
    maxd = df['date'].max()
    cutoff = maxd - pd.Timedelta(days=cutoff_days)
    recent = df[df['date'] >= cutoff]
    a = recent[recent['sw_version'] == sw]['humidity']
    b = recent[recent['sw_version'] != sw]['humidity']
    if len(a) == 0 or len(b) == 0:
        return None
    return float(a.mean()), float(b.mean()), float(a.mean() - b.mean())
 
# -------------------------
# Evidence creation
# -------------------------
def create_evidence_zip(selected_df, title, user, priority, charts=[]):
    evidence_id = "EV-" + uuid4().hex[:8].upper()
    created_on = datetime.datetime.utcnow().isoformat() + "Z"
 
    # CSV
    csv_buf = io.StringIO()
    selected_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode('utf-8')
 
    # Metadata
    metadata = {
        "evidence_id": evidence_id,
        "title": title,
        "created_by": user,
        "priority": priority,
        "created_on": created_on,
        "rows": int(len(selected_df))
    }
    json_bytes = json.dumps(metadata, indent=2).encode('utf-8')
 
    # PDF summary
    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    W, H = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, H - 50, "COMPANY AIRLINES — Compliance Evidence")
    c.setFont("Helvetica", 10)
    c.drawString(40, H - 70, f"Evidence ID: {evidence_id}")
    c.drawString(40, H - 85, f"Title: {title}")
    c.drawString(40, H - 100, f"Created by: {user}    Priority: {priority}")
    c.drawString(40, H - 115, f"Created on: {created_on}")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, H - 140, "Executive Summary")
    c.setFont("Helvetica", 9)
    summary = f"This evidence package contains {len(selected_df)} event rows selected from the Mission Control dashboard to support an Engineering Authorization."
    text_obj = c.beginText(40, H - 155)
    for short in summary.split('. '):
        text_obj.textLine(short.strip())
    c.drawText(text_obj)
    c.showPage()
 
    # Embed chart images (if any)
    for title_img, png_bytes in charts:
        try:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, H - 50, title_img)
            img = ImageReader(io.BytesIO(png_bytes))
            c.drawImage(img, 40, H - 460, width=W - 80, height=400, mask='auto')
            c.showPage()
        except Exception:
            # ignore embed failures for demo
            pass
 
    # Table excerpt (first 30 rows)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, H - 50, "Selected Events (first 30 rows)")
    c.setFont("Helvetica", 8)
    y = H - 70
    cols = list(selected_df.columns)
    header = " | ".join(cols[:8])
    c.drawString(40, y, header[:200])
    y -= 14
    for _, row in selected_df.head(30).iterrows():
        line = " | ".join(str(row[c])[:18] for c in cols[:8])
        c.drawString(40, y, line)
        y -= 12
        if y < 60:
            c.showPage()
            y = H - 60
    c.save()
    pdf_bytes = pdf_buf.getvalue()
 
    # ZIP bundle
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{evidence_id}.csv", csv_bytes)
        zf.writestr(f"{evidence_id}.json", json.dumps(metadata, indent=2).encode('utf-8'))
        zf.writestr(f"{evidence_id}.pdf", pdf_bytes)
    zbuf.seek(0)
    return evidence_id, zbuf.read(), metadata
 
# -------------------------
# Session stores
# -------------------------
if 'audit' not in st.session_state:
    st.session_state['audit'] = []
if 'evidence_repo' not in st.session_state:
    st.session_state['evidence_repo'] = []
 
# -------------------------
# UI styling
# -------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#f8fbff 0%, #ffffff 100%); }
    .kpi { background: #001F3F; color: #fff; padding:12px; border-radius:6px; }
    .panel { background:#fff; border:1px solid #e6eef8; padding:12px; border-radius:6px; }
    .warning { background:#FFD966; padding:8px; border-radius:4px; }
    .badge { display:inline-block; padding:4px 8px; border-radius:4px; background:#e6eef8; }
    </style>
    """,
    unsafe_allow_html=True
)
 
# -------------------------
# App layout & controls
# -------------------------
df = load_df()
st.title("Compliance-Gated Mission Control — Polished Prototype")
st.caption("Three-layer dashboard: Mission Control (leadership) · Engineering Analytics · Compliance-gated actions")
 
# Sidebar controls
st.sidebar.header("Presenter & thresholds")
presenter = st.sidebar.text_input("Presenter name", value="Abdul Shajidh")
cutoff_days = st.sidebar.slider("Cutoff days (before/after)", 7, 45, 15)
priority_pct = st.sidebar.slider("Priority trigger (v2.1 share %)", 5, 80, 30)
min_rows = st.sidebar.number_input("Min rows to package", 1, 50, 2)
 
# Layer 1: Mission Control KPIs
st.subheader("Layer 1 — Mission Control (Executive View)")
k1, k2, k3, k4 = st.columns(4)
total_aids = int(df['aid_count'].sum())
k1.markdown(f"<div class='kpi'><strong>Total AIDs</strong><div style='font-size:20px'>{total_aids}</div></div>", unsafe_allow_html=True)
v21_total = int(df[df['sw_version'] == 'v2.1']['aid_count'].sum())
k2.markdown(f"<div class='kpi'><strong>v2.1 (total)</strong><div style='font-size:20px'>{v21_total}</div></div>", unsafe_allow_html=True)
v21_recent, recent_total, v21_pct = v21_stats(df, cutoff_days=cutoff_days)
k3.markdown(f"<div class='kpi'><strong>v2.1 (last {cutoff_days}d)</strong><div style='font-size:20px'>{v21_recent} ({v21_pct:.0f}%)</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi'><strong>Estimated NFF</strong><div style='font-size:20px'>60-70%</div></div>", unsafe_allow_html=True)
 
st.markdown("---")
 
left, right = st.columns([2, 1], gap="large")
 
with left:
    st.subheader("Layer 2 — Engineering Analytics")
    ds = daily_series(df)
    fig_daily = px.line(ds, x='date', y='aid_count', title="Daily AID Count (fleet)")
    fig_daily.update_layout(template="plotly_white", height=300)
    st.plotly_chart(fig_daily, use_container_width=True)
 
    sw_df, cutoff = sw_before_after(df, cutoff_days=cutoff_days)
    fig_sw = px.bar(
        sw_df.melt(id_vars='sw_version', value_vars=['before', 'after']),
        x='sw_version', y='value', color='variable',
        barmode='group',
        title=f"SW version before vs after (cutoff {cutoff.date()})"
    )
    fig_sw.update_layout(template="plotly_white", height=300)
    st.plotly_chart(fig_sw, use_container_width=True)
 
    v21_series = df[df['sw_version'] == 'v2.1'].groupby(df['date'].dt.date)['aid_count'].sum().reset_index()
    v21_series.columns = ['date', 'aids']
    v21_series['date'] = pd.to_datetime(v21_series['date'])
    fig_v21 = px.line(v21_series, x='date', y='aids', title="v2.1 daily AIDs")
    fig_v21.update_layout(template="plotly_white", height=260)
    st.plotly_chart(fig_v21, use_container_width=True)
 
    st.markdown("### Automated Insight")
    v21_recent, recent_total, v21_pct = v21_stats(df, cutoff_days=cutoff_days)
    insight = f"In the last {cutoff_days} days, SW v2.1 accounts for {v21_recent} AIDs ({v21_pct:.0f}% of recent AIDs)."
    hum = humidity_correlation(df, 'v2.1', cutoff_days=cutoff_days)
    if hum:
        a, b, delta = hum
        insight += f" Average humidity for v2.1 events = {a:.0f}% vs others {b:.0f}% (Δ={delta:.0f}%)."
    st.info(insight)
 
    if v21_pct >= priority_pct:
        st.markdown("<div class='warning'><strong>PRIORITY:</strong> v2.1 correlated spike detected — recommended evidence packaging.</div>", unsafe_allow_h
