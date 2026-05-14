# Disease Outbreak Early Warning System
# Author: Aman Kumar
# Description: A Streamlit dashboard that predicts disease outbreak
#              risk using WHO Ebola surveillance data (2014-2016).
#              Model: Tuned Decision Tree | Recall: 90.6%

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import json
from datetime import datetime
import os

st.set_page_config(
    page_title="Disease Outbreak Early Warning System",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    border-bottom: 3px solid #E05C3A;
    padding-bottom: 0.4rem;
    margin-bottom: 0.5rem !important;
}
h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.3px;
    margin-top: 1.5rem !important;
}
[data-testid="metric-container"] {
    background: #1E1E2E;
    border: 1px solid #2E2E3E;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888 !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] {
    background: #13131F;
    border-right: 1px solid #2E2E3E;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    border-bottom: none !important;
    font-size: 1rem !important;
    color: #ccc;
}
hr {
    border-color: #2E2E3E !important;
    margin: 1.5rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

#  File paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#  Load model and data 
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, 'outbreak_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    with open(os.path.join(BASE_DIR, 'feature_cols.json')) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'ebola_2014_2016_clean.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    return df

model, scaler, feature_cols = load_model()
df = load_data()

#  Title 
st.title("Disease Outbreak Early Warning System")
st.markdown("""
Early detection of disease outbreaks saves lives. This tool analyzes 
recent case trends and flags high-risk situations before they 
become uncontrollable — giving public health officials a 2-week 
head start to respond.
""")
st.divider()

#  Sidebar 
st.sidebar.header("Input Parameters")
st.sidebar.markdown("Enter recent case data for a region:")

country = st.sidebar.selectbox(
    "Select Country",
    options=sorted(df['Country'].unique())
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Recent new case counts:**")

lag1 = st.sidebar.number_input("Cases this reporting period", min_value=0, value=50)
lag2 = st.sidebar.number_input("Cases 1 period ago", min_value=0, value=30)
lag3 = st.sidebar.number_input("Cases 2 periods ago", min_value=0, value=20)

#  Feature calculation
rolling_avg = (lag1 + lag2 + lag3) / 3
rolling_max = max(lag1, lag2, lag3)
rolling_std = np.std([lag1, lag2, lag3])
rate_of_change = ((lag1 - lag2) / lag2) if lag2 > 0 else 0
rate_of_change = float(np.clip(rate_of_change, -10, 10))

now = datetime.now()
month = now.month
week_of_year = now.isocalendar()[1]
season = (month % 12) // 3

country_mapping = {
    'Guinea': 0, 'Italy': 1, 'Liberia': 2, 'Mali': 3,
    'Nigeria': 4, 'Senegal': 5, 'Sierra Leone': 6,
    'Spain': 7, 'United Kingdom': 8, 'United States of America': 9
}
country_code = country_mapping.get(country, 0)

input_data = pd.DataFrame([[
    lag1, lag2, lag3,
    rolling_avg, rolling_max, rolling_std,
    rate_of_change, month, week_of_year,
    season, country_code
]], columns=feature_cols)

input_scaled = scaler.transform(input_data)
probability = model.predict_proba(input_scaled)[0][1]

#  Risk level 
if probability < 0.3:
    risk_level = "LOW"
    banner_bg = "#1A3A2A"
    banner_border = "#2ECC71"
    banner_text = "#2ECC71"
elif probability < 0.6:
    risk_level = "MODERATE"
    banner_bg = "#3A2E1A"
    banner_border = "#F39C12"
    banner_text = "#F39C12"
else:
    risk_level = "HIGH"
    banner_bg = "#3A1A1A"
    banner_border = "#E05C3A"
    banner_text = "#E05C3A"

#  Risk display 
st.subheader("Outbreak Risk Assessment")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Selected Country", value=country)
with col2:
    st.metric(label="Outbreak Probability", value=f"{probability:.1%}")
with col3:
    st.metric(label="Risk Level", value=risk_level)

st.markdown(f"""
<div style='
    background: {banner_bg};
    border: 1px solid {banner_border};
    border-left: 4px solid {banner_border};
    border-radius: 10px;
    padding: 18px 24px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 12px;
'>
    <span style='color:{banner_text};font-size:1.8rem;line-height:1'>●</span>
    <div>
        <div style='color:{banner_text};font-size:0.7rem;text-transform:uppercase;
                    letter-spacing:0.1em;font-weight:600;margin-bottom:2px'>
            Risk Assessment
        </div>
        <div style='color:{banner_text};font-size:1.3rem;font-weight:700'>
            {risk_level} OUTBREAK RISK — {probability:.1%} probability
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

#  Historical trend 
st.subheader(f"Historical Case Trend — {country}")

country_data = df[df['Country'] == country].copy()

fig = px.line(
    country_data,
    x='Date',
    y='New_Cases',
    title=f'Weekly New Cases — {country}',
    labels={'New_Cases': 'New Cases', 'Date': 'Date'},
    color_discrete_sequence=['#E05C3A']
)
fig.update_layout(
    height=350,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='#aaa',
    xaxis=dict(gridcolor='#2E2E3E'),
    yaxis=dict(gridcolor='#2E2E3E')
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Feature importance 
st.subheader("What's driving this prediction?")

feat_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

fig2 = px.bar(
    feat_df,
    x='Importance',
    y='Feature',
    orientation='h',
    title='Feature Importance — Model explanation',
    color='Importance',
    color_continuous_scale='Oranges'
)
fig2.update_layout(
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='#aaa',
    xaxis=dict(gridcolor='#2E2E3E'),
    yaxis=dict(gridcolor='#2E2E3E')
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

#  Input summary 
st.subheader("Input Summary")

st.markdown(f"""
<div style='
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 8px;
'>
    <div style='background:#1E1E2E;border:1px solid #2E2E3E;
                border-radius:12px;padding:16px 20px'>
        <div style='font-size:0.7rem;text-transform:uppercase;
                    letter-spacing:0.08em;color:#888;margin-bottom:10px'>
            Case Inputs
        </div>
        <div style='font-size:0.95rem;line-height:2;color:#ddd'>
            Current period &nbsp;<strong style='color:white'>{lag1} cases</strong><br>
            Previous period &nbsp;<strong style='color:white'>{lag2} cases</strong><br>
            2 periods ago &nbsp;<strong style='color:white'>{lag3} cases</strong>
        </div>
    </div>
    <div style='background:#1E1E2E;border:1px solid #2E2E3E;
                border-radius:12px;padding:16px 20px'>
        <div style='font-size:0.7rem;text-transform:uppercase;
                    letter-spacing:0.08em;color:#888;margin-bottom:10px'>
            Derived Features
        </div>
        <div style='font-size:0.95rem;line-height:2;color:#ddd'>
            Rolling average &nbsp;<strong style='color:white'>{rolling_avg:.1f}</strong><br>
            Rate of change &nbsp;<strong style='color:white'>{rate_of_change:.2f}</strong><br>
            Std deviation &nbsp;<strong style='color:white'>{rolling_std:.1f}</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption("Built by Aman Kumar | Model: Tuned Decision Tree | Data: WHO Ebola Surveillance 2014–2016")
