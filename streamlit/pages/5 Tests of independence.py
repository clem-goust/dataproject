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

# Test du chi-2 pour les tests d'indépendance de variables catégorielles 
from scipy.stats import chi2_contingency

# Analyse de la variance (ANOVA) pour les tests d'indépendance de variable catégorielle & numérique
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Chargement des fichiers:

df_prospects_metrics = pd.read_csv("streamlit\output_streamlit\prospects_metrics.csv", index_col=0)

# Titre de la page

st.title("Garanteo | Tests of Independence")
with st.expander(label="## See comments"):
    st.write(f'''
            Here we test the mutual independence and (if need be) the correlation of the variables used for machine learning. We want to avoid including multiple variables with equivalent meanings. 
            '''
    )

# Variables numériques:

df_prospects_num = df_prospects_metrics[[
    'age',
    'session_count', 
    'avg_days_btw', 
    'sum_duration_in_min',
    'CTC',
    'click',
    'sessions_before_first_ctc',
    'clicks_before_first_ctc',
    'time_online_before_first_ctc_in_sec', 
    'days_to_first_ctc'
]]

df_prospects_num = df_prospects_num.astype(float)

# Variables catégorielles:

df_prospects_cat = df_prospects_metrics[[
    'is_client',
    'gender',
    'device_type',
    'device_browser',
    'device_operating_system',
    'device_language',
    'country',
    'user_preference', 
    'user_behavior'
]]

df_prospects_cat = df_prospects_cat.astype('category')

# df avec seulement les variables à tester:

df = pd.concat([df_prospects_num, df_prospects_cat], axis=1)

# Création du multiselect en sidebar

variables = st.sidebar.multiselect(label="Select two variables to be tested", options=list(df.columns), max_selections=2)

# Dictionnaire de description des variables:

dict_describe = {
    'is_client':"Whether the prospect is client or not. We will try to predict it with a ML model.",
    'age':"Age of prospect. Mean around 80 yo. Max around 140 yo. Which is quite high, in both cases...",
    'session_count': "Total number of sessions on Garanteo's website" , 
    'avg_days_btw': "Average number of days bewteen two consecutive sessions",
    'sum_duration_in_min': "Cumulated time spent on Garanteo's website",
    'CTC':"Total number of click-to-calls (callback demands) on Garanteo's website",
    'click':"Total number of clicks on Garanteo's website",
    'sessions_before_first_ctc': "Number of sessions before the session of the first click-to-call",
    'clicks_before_first_ctc':"Total number of clicks before first click-to-call",
    'time_online_before_first_ctc_in_sec':"Cumulated time spent on Garanteo's website until first click-to-call", 
    'days_to_first_ctc':"Total number of days between first session and the day of the first click-to-call",
    'gender':"Gender of prospect",
    'device_type':"Desktop or mobile",
    'device_browser':"Browser used to reach Garanteo's website",
    'device_operating_system':"Windows for desktops, IOS or Android for mobiles",
    'device_language':"Language of the device used to connect to Garanteo's website. May differ from Country",
    'country':"Country where the connection was made",
    'user_preference':"Page preference if any. Based on prospect's web trafic data", 
    'user_behavior':"Connections habits (weekday/weekend, day/night) if any. Based on prospect's web trafic data"
}

#  Setup de la page

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True, height=520):
        try:
            st.subheader(variables[0].replace('_',' ').capitalize())
            with st.container(border=False, height=80):
                st.write(dict_describe[variables[0]])
            if df[variables[0]].dtype == float:
                # Si la variable est numérique, on fait un histplot
                with st.container(height=250, border=False):
                    g6a, ax6a = plt.subplots() 
                    sns.histplot(
                        data=df, 
                        x=df[variables[0]],
                        shrink = 0.8,
                        ax=ax6a, 
                        color='steelblue',
                        edgecolor=None
                    )
                    ax6a.spines['top'].set_visible(False)
                    ax6a.spines['right'].set_visible(False)
                    ax6a.set(xlabel=f"\n{variables[0].replace('_',' ').capitalize()}", ylabel=f"Count\n")
                    st.pyplot(g6a, use_container_width=True)
                st.write('_Numerical variable_')

            elif df[variables[0]].dtype == 'category':
                # Si la variable est catégorielle, on fait un catplot
                with st.container(height=250, border=False):
                    g6a = sns.catplot(
                        data=df, 
                        x=df[variables[0]],
                        kind='count',
                        color='steelblue',
                        height=4,
                        aspect=1.5
                    )
                    g6a.set_axis_labels(f"\n{variables[0].replace('_',' ').capitalize()}", "Count\n")
                    plt.xticks(rotation=40)
                    st.pyplot(g6a, use_container_width=True)
                st.write('_Categorical variable_')
            
            else:
                st.write("There must be a problem with variable selection")
        except:
            st.subheader("No variable selected")
            st.image("streamlit/images/data.jpg", use_column_width=True)

with col2:
    with st.container(border=True, height=520):
        try:
            st.subheader(variables[1].replace('_',' ').capitalize())
            with st.container(border=False, height=80):
                st.write(dict_describe[variables[1]])
            if df[variables[1]].dtype == float:
                # Si la variable est numérique, on fait un histplot
                with st.container(height=250, border=False):
                    g6b, ax6b = plt.subplots() 
                    sns.histplot(
                        data=df, 
                        x=df[variables[1]],
                        shrink = 0.8,
                        ax=ax6b, 
                        color='steelblue',
                        edgecolor=None
                    )
                    ax6b.spines['top'].set_visible(False)
                    ax6b.spines['right'].set_visible(False)
                    ax6b.set(xlabel=f"\n{variables[1].replace('_',' ').capitalize()}", ylabel=f"Count\n")
                    st.pyplot(g6b, use_container_width=True)
                st.write('_Numerical variable_')

            elif df[variables[1]].dtype == 'category':
                # Si la variable est catégorielle, on fait un catplot
                with st.container(height=250, border=False):
                    g6b = sns.catplot(
                        data=df, 
                        x=df[variables[1]],
                        kind='count',
                        color='steelblue',
                        height=4,
                        aspect=1.5
                    )
                    g6b.set_axis_labels(f"\n{variables[1].replace('_',' ').capitalize()}", "Count\n")
                    plt.xticks(rotation=40)
                    st.pyplot(g6b, use_container_width=True)
                st.write('_Categorical variable_')
            
            else:
                st.write("There must be a problem with variable selection")
        except:
            st.subheader("No variable selected")
            st.image("streamlit/images/data.jpg", use_column_width=True)

with st.container(border=True):
    try:
        if (df[variables[0]].dtype, df[variables[1]].dtype)  == ('category', 'category'):
            
            # Table de contingence et test du khi carré
            crosstab = pd.crosstab(df[variables[0]], df[variables[1]])
            chi2, p, dof, expected = chi2_contingency(crosstab)

            # Nombre total d'observations
            n = crosstab.sum().sum()

            # Taille minimale entre les dimensions du tableau de contingence
            k = min(crosstab.shape)

            # Calcul du V de Cramer
            cramer_v = np.sqrt(chi2 / (n * (k - 1)))

            # Si la p-valeur est inférieure à 5%, alors on peut rejeter l'hypothèse d'indépendance des deux variables
            if p<0.05:
                st.subheader(":x: Not independent")
                st.write(f'''
                Method used: chi-squared test\n
                    p-value: {round(p,3)}\n
                    Degrees of freedom: {dof}\n
                    Cramer's V: {round(cramer_v,3)}
                ''')
            else:
                st.subheader(":white_check_mark: Independent")
                st.write(f'''
                Method used: chi-squared test, 95% confidence\n
                _p-value: {round(p,3)}_\n
                _Degrees of freedom: {dof}_''')

        elif (df[variables[0]].dtype, df[variables[1]].dtype)  == (float, float):
            pearson = np.corrcoef(df[variables[0]], df[variables[1]])[0][1]
            if (pearson <=1) and (pearson >= 0.8):
                st.subheader("Strong positive linear correlation")
            elif (pearson < 0.8) and (pearson >= 0.5):
                st.subheader("Moderate positive linear correlation")
            elif (pearson < 0.5) and (pearson >= 0.25):
                st.subheader("Low to no positive linear correlation")
            elif (pearson < 0.25) and (pearson >= -0.25):
                st.subheader("No linear correlation")
            elif (pearson < -0.25) and (pearson >= -0.50):
                st.subheader("Low to no negative linear correlation")
            elif (pearson < 0.50) and (pearson >= -0.8):
                st.subheader("Moderate negative linear correlation")
            elif (pearson < -0.8) and (pearson >= -1):
                st.subheader("Strong negative linear correlation")
            else:
                st.subheader("There must be an issue with data...")
            st.write(f'''
            Method used: Pearson correlation coefficient\n
                Coefficient: {round(pearson,3)}
            ''')

        elif (df[variables[0]].dtype, df[variables[1]].dtype)  == ('category', float):
            
            # Création d'un modèle OLS (Ordinary Least Squares)
            formula = f'{variables[1]} ~ C({variables[0]})'  # C() indique que c'est une variable catégorielle
            model = smf.ols(formula, data=df).fit()
            anova_results = anova_lm(model, typ=2)
            
            # Retourner la p-value associée à la variable catégorielle
            p = anova_results["PR(>F)"][0]  # La p-value se trouve dans la première ligne de l'ANOVA

            # Calcul du R²
            r2 = model.rsquared

            if p<0.05:
                st.subheader(":x: Not independent")
                st.write(f'''
                Method used: ANOVA (anaysis of variance), 95% confidence\n
                    p-value: {round(p,3)}\n
                    R-squared: {round(r2,3)}
                ''')
            else:
                st.subheader(":white_check_mark: Independent")
                st.write(f'''
                Method used: ANOVA (anaysis of variance), 95% confidence\n
                    p-value: {round(p,3)}\n
                ''')        
        
        elif (df[variables[0]].dtype, df[variables[1]].dtype)  == (float, 'category'):
            
            # Création d'un modèle OLS (Ordinary Least Squares)
            formula = f'{variables[0]} ~ C({variables[1]})'  # C() indique que c'est une variable catégorielle
            model = smf.ols(formula, data=df).fit()
            anova_results = anova_lm(model, typ=2)
            
            # Retourner la p-value associée à la variable catégorielle
            p = anova_results["PR(>F)"][0]  # La p-value se trouve dans la première ligne de l'ANOVA

            # Calcul du R²
            r2 = model.rsquared

            if p<0.05:
                st.subheader(":x: Not independent")
                st.write(f'''
                Method used: ANOVA (anaysis of variance)\n
                    p-value: {round(p,3)}\n
                    R-squared: {round(r2,3)}
                ''')
            else:
                st.subheader(":white_check_mark: Independent")
                st.write(f'''
                Method used: ANOVA (anaysis of variance), 95% confidence\n
                    p-value: {round(p,3)}\n
                ''')
        
        else:
            st.subheader("There must an issue with data...")
    except:
        st.subheader("Please select two variables for independence testing")