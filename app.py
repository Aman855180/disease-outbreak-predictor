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

# Full path of project folder
BASE_DIR = r'C:\Outbreak_project'

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

#  Calling  functions to actually load everything
model, scaler, feature_cols = load_model()
df = load_data()
# Title 
st.title(" Disease Outbreak Early Warning System")
st.markdown("""
Early detection of disease outbreaks saves lives. This tool analyzes 
recent case trends and flags high-risk situations before they 
become uncontrollable by giving public health officials a 2-week 
head start to respond.
""")
st.divider()

# Sidebar inputs 
st.sidebar.header(" Input Parameters")
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

# Feature calculation
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

# Risk level 
if probability < 0.3:
    risk_level, risk_color = "LOW", "green"
elif probability < 0.6:
    risk_level, risk_color = "MODERATE", "orange" 
else:
    risk_level, risk_color = "HIGH", "red"

#  Risk display 
st.subheader(" Outbreak Risk Assessment")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Selected Country", value=country)
with col2:
    st.metric(label="Outbreak Probability", value=f"{probability:.1%}")
with col3:
    st.metric(label="Risk Level", value=f" {risk_level}")

st.markdown(f"""
<div style='background-color:{risk_color};
            padding:20px;
            border-radius:10px;
            text-align:center;
            color:white;
            font-size:24px;
            font-weight:bold;
            margin:10px 0'>
     {risk_level} OUTBREAK RISK — {probability:.1%} probability
</div>
""", unsafe_allow_html=True)

st.divider()

# Historical trend 
st.subheader(f" Historical Case Trend — {country}")

country_data = df[df['Country'] == country].copy()

fig = px.line(
    country_data,
    x='Date',
    y='New_Cases',
    title=f'Weekly New Cases — {country}',
    labels={'New_Cases': 'New Cases', 'Date': 'Date'},
    color_discrete_sequence=['steelblue']
)
fig.update_layout(height=350)
st.plotly_chart(fig, width='stretch')

st.divider()

#  Feature importance
st.subheader(" What's driving this prediction?")

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
    color_continuous_scale='Blues'
)
fig2.update_layout(height=400)
st.plotly_chart(fig2, width='stretch')

st.divider()

#  Input summary
st.subheader(" Input Summary")

s1, s2 = st.columns(2)
with s1:
    st.markdown("**Case inputs:**")
    st.write(f"- Current period: {lag1} cases")
    st.write(f"- Previous period: {lag2} cases")
    st.write(f"- 2 periods ago: {lag3} cases")

with s2:
    st.markdown("**Derived features:**")
    st.write(f"- Rolling average: {rolling_avg:.1f}")
    st.write(f"- Rate of change: {rate_of_change:.2f}")
    st.write(f"- Rolling std dev: {rolling_std:.1f}")

st.caption("Built by Aman Kumar | Model: Tuned Decision Tree | Data: WHO Ebola Surveillance 2014–2016")