## Initialisation

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

df_users_metrics = pd.read_csv(r"streamlit\output_streamlit\users_metrics.csv", index_col=0)
df_prospects_metrics = pd.read_csv(r"streamlit\output_streamlit\prospects_metrics.csv", index_col=0)
df_users_segments = pd.read_csv(r"streamlit\output_streamlit\users_segments.csv", index_col=0)

# Initialisation des tables

df_users = df_users_metrics.merge(df_users_segments, on='user_id')
df_prospects = df_prospects_metrics
df_leads = df_prospects[(df_prospects['is_presented_prospect']==0)&(df_prospects['is_client']==0)]
df_clients = df_prospects[df_prospects['is_client']==1]

# Sélection du df

view_options = ['Users', 'Prospects', 'Leads', 'Clients']
view = st.sidebar.selectbox("Select a population :", view_options)
df1 = globals()[f'df_{view.lower()}']

st.title(f'Garanteo | Overview of {view}')

tab_a, tab_b, tab_c, tab_d, tab_e, tab_f, tab_g, tab_h = st.tabs([
    ":iphone: OS", 
    ":compass: Browser", 
    ":speaking_head_in_silhouette: Language", 
    ":globe_with_meridians: Country", 
    ":medal: Page pref.", 
    ":crescent_moon: Connection habit", 
    ":restroom: Gender", 
    ":baby: Age"
])

with tab_a:

    # Graph des OS

    g1a = sns.catplot(
        data=df1, 
        x=df1['device_operating_system'].str.capitalize(),
        order=['Windows','Ios','Android'],
        kind='count',
        hue=df1['device_type'].rename('Device type').str.capitalize(),
        hue_order=['Desktop', 'Mobile'], 
        palette="mako",
        height=7,
        aspect=1.3,
        legend='brief',
        legend_out=False
    )

    g1a.set_axis_labels("\nOperating System", f"{view} count\n", fontsize=20)
    g1a.set_xticklabels(fontsize=20)
    g1a.set_yticklabels(fontsize=20)

    plt.legend(
        title='Device type\n',
        title_fontsize=20,
        bbox_to_anchor=(1.05, 1), 
        loc=2, 
        borderaxespad=0,
        markerscale=1, 
        fontsize=20, 
        frameon=False
    )

    st.subheader(f"{view}' operating systems")
    st.pyplot(g1a)

with tab_b:

    # Graph des navigateurs
    
    g1b = sns.catplot(
        data=df1, 
        x=df1['device_browser'].str.capitalize(),
        kind='count',
        height=7,
        aspect=1.7,
        color='steelblue'
    )

    g1b.set_axis_labels("\nBrowser", f"{view} count\n", fontsize=20)
    g1b.set_xticklabels(fontsize=20)
    g1b.set_yticklabels(fontsize=20)
    st.subheader(f"{view}' browsers")
    st.pyplot(g1b)

with tab_c:

    # Graph des langues

    g1c = sns.catplot(
        data=df1, 
        x=df1['device_language'],
        order=['FR', 'EN', 'Other'],
        kind='count',
        height=7,
        aspect=1.7,
        color='steelblue'
    )

    g1c.set_axis_labels("\nDevice language", f"{view} count\n", fontsize=20)
    g1c.set_xticklabels(fontsize=20)
    g1c.set_yticklabels(fontsize=20)
    st.subheader(f"{view}' device languages")
    st.pyplot(g1c)

with tab_d:

    # Graph des pays

    g1d = sns.catplot(
        data=df1, 
        x=df1['country'].str.capitalize(),
        kind='count',
        height=7,
        aspect=1.7,
        color='steelblue'
    )

    g1d.set_axis_labels("\nCountry", f"{view} count\n", fontsize=20)
    g1d.set_xticklabels(fontsize=20)
    g1d.set_yticklabels(fontsize=20)
    st.subheader(f"{view}' countries")
    st.pyplot(g1d)

with tab_e:

    # Graph des préférences

    g1e = sns.catplot(
        data=df1, 
        x=df1['user_preference'],
        order=['Multipages', 'Page 1', 'Page 2', 'Page 3', 'Page 4', 'Page 5', 'Page 6', 'Peu intéressé'],
        kind='count',
        height=7,
        aspect=1.7,
        color='steelblue'
    )

    g1e.set_axis_labels("\nPage preference", f"{view} count\n", fontsize=20)
    g1e.set_xticklabels(fontsize=20)
    g1e.set_yticklabels(fontsize=20)

    st.subheader(f"{view}' webpage preferences")
    plt.xticks(rotation=40)
    st.pyplot(g1e)

with tab_f:

    # Graph des habitudes de connexion

    g1f = sns.catplot(
        data=df1, 
        y=df1['user_behavior'].str.capitalize(),
        order=['Weekday', 'Weekday daytime', 'Weekday nighttime', 'Weekend', 'Weekend daytime', 'Weekend nighttime', 'Daytime', 'Nighttime', 'No preference'],
        kind='count',
        height=7,
        aspect=1.7,
        color='steelblue'
    )

    g1f.set_axis_labels(f"\n{view} count", f"Connection behavior\n", fontsize=20)
    g1f.set_xticklabels(fontsize=20)
    g1f.set_yticklabels(fontsize=20)

    st.subheader(f"{view}' connection behaviors")
    plt.xticks(rotation=90)
    st.pyplot(g1f)

with tab_g:

    # Graph des genres

    if view == 'Users':
        with st.expander("Gender data: not available for Users"):
            st.write(f'''
                Gender data is not available for Users. It is only available for Prospects, Leads & Clients, as it is collected when a User proceeds to a Click-to-Call (callback demand), which triggers the Prospect onboarding process.
            ''')
        
        col1, col2, col3 = st.columns([1,5,1])
        with col1:
            st.write()
        with col2:
            st.image("streamlit/images/data.jpg", use_column_width=True)
        with col3:
            st.write()

    else:
        g1h = sns.catplot(
            data=df1, 
            x=df1['gender'],
            kind='count',
            height=7,
            aspect=1.7,
            color='steelblue'
        )

        g1h.set_axis_labels("\nGender", f"{view} count\n", fontsize=20)
        g1h.set_xticklabels(fontsize=20)
        g1h.set_yticklabels(fontsize=20)

        st.subheader(f"{view}' genders")
        st.pyplot(g1h)

with tab_h:

    # Graph des âges

    if view == 'Users':

        with st.expander("Age data: not available for Users"):
            st.write(f'''
                Age data is not available for Users. It is only available for Prospects, Leads & Clients, as it is collected when a User proceeds to a Click-to-Call (callback demand), which triggers the Prospect onboarding process.
            ''')

        col1, col2, col3 = st.columns([1,5,1])
        with col1:
            st.write()
        with col2:
            st.image("streamlit/images/data.jpg", use_column_width=True)
        with col3:
            st.write()

    else:
        g1g, ax = plt.subplots() 
        
        sns.histplot(
            data=df1, 
            y=df1['age'],
            binwidth=20,
            shrink = 0.8,
            ax=ax, 
            color='steelblue',
            edgecolor=None
        )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set(ylabel="Age", xlabel=f"{view} count")
        st.subheader(f"{view}' ages")

        st.pyplot(g1g)