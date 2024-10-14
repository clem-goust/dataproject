# Initialisation

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import datetime as dt
import warnings
import networkx as nx
import streamlit as st

warnings.simplefilter('ignore')

# Chargement des fichiers

df_prospects_metrics = pd.read_csv("streamlit\output_streamlit\prospects_metrics.csv", index_col=0)
df_scoring = pd.read_csv("streamlit\output_streamlit\scoring.csv", index_col=0)

df = pd.merge(df_prospects_metrics, df_scoring, on='user_id')

# Choix du rating

st.sidebar.write("Select a scoring category")
select = st.sidebar.feedback("stars")

if select == None:
    selected = 4
else:
    selected = select

df = df[df['score']==(selected+1)]

view = st.sidebar.selectbox("Select a lead :", df.sort_values('last_ctc_date', ascending=True)['user_id'])

gender = df[df['user_id']==view]['gender'].values[0]
age = df[df['user_id']==view]['age'].values[0]
page_pref = df[df['user_id']==view]['user_preference'].values[0]
date = df[df['user_id']==view]['last_ctc_date'].values[0]
behavior = df[df['user_id']==view]['user_behavior'].values[0]
language = df[df['user_id']==view]['device_language'].values[0]
country = df[df['user_id']==view]['country'].values[0]


# Titre

st.title("Garanteo | Leads Dashboard")
st.subheader(f"Selected lead: {view}")
st.subheader(":star:"*(selected+1))
st.divider()

# Colonnes

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(height=150, border=True):
        st.image(f"streamlit/images/{gender}.jpg")
    with st.container(height=150, border=True):
        st.write("Page pref.")
        st.subheader(f"{page_pref}")

with col2:
    with st.container(height=150, border=True):
        st.write("Gender")
        st.subheader(f"{gender}")
    with st.container(height=150, border=True):
        st.write("Connects")
        st.subheader(f"{behavior}")

with col3:
    with st.container(height=150, border=True):
        st.write("Age")
        st.subheader(f"{age}")
    with st.container(height=150, border=True):
        st.write("Language")
        st.subheader(f"{language}")

with col4:
    with st.container(height=150, border=True):
        st.write("Last CTC")
        st.subheader(f"{date}")
    with st.container(height=150, border=True):
        st.write("Country")
        st.subheader(f"{country}")

with st.expander("See comments"):
    st.write(f'''
        The Leads Dashboard is a tool for Garanteo's Sales teams. In a single look, they will find here all the information they need before a lead call.\n
        - The **score** measures the probability that the prospect becomes a client upon a call. It was derived from a machine learning model (see *Machine learning* page).
        - Among the best-rated leads, Garanteo should prioritize the leads with older callback demands (**Last CTC**)
    ''')