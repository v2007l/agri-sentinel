import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load predictions
df = pd.read_csv('data/processed/tn_predictions.csv')

# Page config
st.set_page_config(
    page_title="AGRI-SENTINEL",
    page_icon="🛰️",
    layout="wide"
)

# Header
st.title("🛰️ AGRI-SENTINEL")
st.subheader("India's First Agricultural Crisis Intelligence System")
st.markdown("---")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
critical = len(df[df['alert_level'] == '🚨 CRITICAL'])
high = len(df[df['alert_level'] == '⚠️  HIGH'])
moderate = len(df[df['alert_level'] == '🟡 MODERATE'])
safe = len(df[df['alert_level'] == '✅ SAFE'])

col1.metric("🚨 Critical", critical, "Immediate action!")
col2.metric("⚠️ High Risk", high, "Monitor closely")
col3.metric("🟡 Moderate", moderate, "Watch carefully")
col4.metric("✅ Safe", safe, "Normal conditions")

st.markdown("---")

# Year filter
col_f1, col_f2 = st.columns(2)
year = col_f1.selectbox("Select Year", sorted(df['year'].unique(), reverse=True))
month = col_f2.selectbox("Select Month", {1:'January',4:'April',7:'July',10:'October'}.items(),
                          format_func=lambda x: x[1])

# Filter data
filtered = df[(df['year'] == year) & (df['month'] == month[0])].sort_values(
    'predicted_stress', ascending=True)

st.subheader(f"📊 District Stress Levels — {year}")

# Color map
color_map = {
    '🚨 CRITICAL': '#E24B4A',
    '⚠️  HIGH':    '#EF9F27',
    '🟡 MODERATE': '#F9E2AF',
    '✅ SAFE':     '#A6E3A1'
}

# Bar chart
fig = px.bar(
    filtered,
    x='predicted_stress',
    y='district',
    color='alert_level',
    color_discrete_map=color_map,
    orientation='h',
    title=f'Compound Stress Score by District',
    labels={'predicted_stress': 'Stress Score', 'district': 'District'}
)
fig.update_layout(height=700, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Trend chart
st.subheader("📈 District Stress Trend — All Years")
district = st.selectbox("Select District", sorted(df['district'].unique()))
trend = df[df['district'] == district].sort_values(['year','month'])

fig2 = px.line(
    trend,
    x='year',
    y='predicted_stress',
    color='month',
    markers=True,
    title=f'{district} — Stress Trend 2020-2023',
    labels={'predicted_stress': 'Stress Score', 'year': 'Year'}
)
fig2.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Critical threshold")
fig2.add_hline(y=25, line_dash="dash", line_color="orange", annotation_text="High threshold")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("🛰️ AGRI-SENTINEL — Powered by Sentinel-2 + CHIRPS Satellite Data + Random Forest AI")