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

st.title('Garanteo | Web Trafic')

df_campaigns_metrics = pd.read_csv("streamlit\output_streamlit\campaigns_metrics.csv", index_col=0)
df_sessions = pd.read_csv("streamlit\output_streamlit\sessions.csv", index_col=0)
df_events = pd.read_csv("streamlit\output_streamlit\events.csv", index_col=0)
df_campaigns_pagefocus = pd.read_csv("streamlit\output_streamlit\campaigns_pagefocus.csv", index_col=0)

df_left = df_sessions
df_right = df_events[['event_id','session_id','event_timestamp','event_type','page','referrer','referrer_summary','medium','campaign_id','user_id']]
df4a = pd.merge(left = df_left, right = df_right, how = 'left', on = ['session_id', 'user_id'])\
                .sort_values(['user_id', 'event_timestamp'])\
                .reset_index(drop=True)

df_campaigns_pagefocus = df_campaigns_pagefocus.drop(columns=['rebound_rate','ctc_rate']) 

# On extrait la liste des campaign_id
campaigns = list(set(df_campaigns_metrics['campaign_id']))
campaigns.sort()

# On extrait la liste des pages
pages = list(set(df_events['page']))
pages.sort()

# On initialise le dictionnaire avec les visites pour chaque campagne & pour chaque page
dico_campaigns = {}
for campaign in campaigns:
    subdf4a = df4a[df4a['session_id'].isin(df4a[df4a['campaign_id']==campaign]['session_id'])][['campaign_id','page']]
    subdico = {}
    for page in pages:
        subdico[page] = subdf4a[subdf4a['page']==page].shape[0]
    dico_campaigns[campaign] = subdico

# Choix de la campagne
campaigns_options = ["All"] + campaigns 
campaign = st.sidebar.selectbox("Select a campaign", campaigns_options)

if campaign == 'All':
    df4b = df_campaigns_pagefocus\
                .drop(columns='campaign_id')\
                .groupby('page')\
                .agg({
                    'visit_count':'sum',
                    'cumul_time_in_sec':'sum',
                    'ctc_count':'sum',
                    'rebound_count':'sum',
                    'redirect_count':'sum'
                })\
                .reset_index()
else:
    df4b = df_campaigns_pagefocus[df_campaigns_pagefocus['campaign_id']==campaign]\
                .drop(columns='campaign_id')
df4b['rebound_rate'] = df4b['rebound_count']/df4b['redirect_count']
df4b['ctc_rate'] = df4b['ctc_count']/df4b['visit_count']

tab_a, tab_b = st.tabs([
    ":spider_web: Transitions", 
    ":trophy: Performance"
])

with tab_a:
    @st.cache_data
    def plot_my_graph(df=df4a, campaign_id='All'):
        
        # En fonction de la campagne choisie, on filtre la table et le dictionnaire de visites sur lesquels on va travailler
        if campaign_id != 'All':
            
            # On filtre df_rebound
            subdf = df[df['session_id']\
                .isin(df[df['campaign_id']==campaign_id]['session_id'])\
                ].reset_index(drop=True)

            # On crée la table des chemins
            df_sessions_path = subdf[['user_id', 'session_started_at', 'page']]\
                .groupby(['user_id','session_started_at'])\
                ['page']\
                .apply(list)\
                .reset_index()
            
            # On filtre le dico
            dico_visits = dico_campaigns[campaign_id]

        # Sinon, on prend par défaut toutes les campagnes
        else:
            df_sessions_path = df[['user_id', 'session_started_at', 'page']]\
                .groupby(['user_id','session_started_at'])\
                ['page']\
                .apply(list)\
                .reset_index()
            
            # On crée le dico avec toutes les campagnes
            subdico = {}
            for page in pages:
                subdico[page] = df[df['page']==page].shape[0]
            dico_visits = subdico   
                    
        transitions = []
        
        for page_list in df_sessions_path['page']:
            if len(page_list) == 1:
                pass
            else:
                for i in range(len(page_list) - 1):
                    transitions.append((page_list[i], page_list[i+1]))

        weighted_transitions\
            = [(pd.Series(transitions).value_counts().index[i][0], 
            pd.Series(transitions).value_counts().index[i][1], 
            pd.Series(transitions).value_counts()[i]) for i in range(pd.Series(transitions).value_counts().shape[0])]

        # Créer un graphe orienté
        G = nx.DiGraph()

        # Ajouter les nœuds (pages)
        G.add_nodes_from(pages)

        # Ajouter les arêtes pondérées (transitions entre les pages)
        for u, v, weight in weighted_transitions:
            G.add_edge(u, v, weight=weight)
            
        # Positionner les nœuds pour la visualisation
        pos = nx.spring_layout(G)
        
        # Couleurs des nœuds basées sur les visites (avec une échelle manuelle)
        # Définir manuellement les valeurs min et max pour l'échelle de couleurs
        min_visits, max_visits = 0, max(dico_visits.values())  
        node_colors = [dico_visits[page] for page in pages]

        # Taille des nœuds basée sur le nombre de visites
        node_sizes = [dico_visits[page] * (2000/max_visits)  for page in pages]  # Multiplier par (200/max_visits) pour régler la taille
        
        # Normaliser les valeurs des visites pour les mapper correctement aux couleurs
        norm = mpl.colors.Normalize(vmin=min_visits, vmax=max_visits)
        cmap = plt.cm.coolwarm
        
        # Couleurs des arêtes avec une échelle manuelle
        min_weight, max_weight = 0, max([weight for u, v, weight in weighted_transitions])
        edge_colors = [G[u][v]['weight'] for u, v in G.edges()]

        # Épaisseur des arêtes en fonction des poids (fréquence des transitions)
        edge_weights = [G[u][v]['weight'] / (0.4*max_weight) for u, v in G.edges()] # Diviser par (0.5*max_weight) pour régler l'épaisseur
        
        # Normaliser les poids des arêtes pour correspondre à une échelle de couleurs
        edge_norm = mpl.colors.Normalize(vmin=min_weight, vmax=max_weight)
        edge_cmap = plt.cm.Greys

        # Taille du graphe 
        fig, ax = plt.subplots(figsize=(12, 8))

        # Dessiner les nœuds
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap, alpha=1, vmin=min_visits, vmax=max_visits, ax=ax)

        # Dessiner les étiquettes des nœuds
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)

        # Dessiner les arêtes avec des épaisseurs et couleurs personnalisées
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors, edge_cmap=edge_cmap, edge_vmin=min_weight, edge_vmax=max_weight, alpha=0.6, ax=ax)

        # Ajouter les étiquettes des arêtes (les poids)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        
        # Ajouter une légende pour la couleur des nœuds (basée sur les visites)
        sm_nodes = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm_nodes.set_array([])  # Nécessaire pour créer une légende
        cbar_nodes = plt.colorbar(sm_nodes, ax=ax, label='Visits count')

        # Ajouter une légende pour la couleur des arêtes (basée sur les transitions)
        sm_edges = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
        sm_edges.set_array([])  # Nécessaire pour créer une légende
        cbar_edges = plt.colorbar(sm_edges, ax=ax, label='Transitions frequencies')

        st.header(f"Page transitions")
        st.subheader(f"Campaign : _{campaign_id}_")
        with st.expander("See comments"):
            st.write('''
                This graph's purpose is to profile the typical path (if any) of those users reached by a given campaign.\n
                For each campaign, we check how often users visited each of the 6 webpages (nodes colors).\n
                Edges width, colors and tags tell us about the frequencies of the transitions between the 6 pages.
            ''')
        st.pyplot(fig)

    plot_my_graph(campaign_id=campaign)
    

with tab_b:
    # Taux de rebond par page vs taux de CTC par page

    minsize = df4b['visit_count'].min()
    maxsize = df4b['visit_count'].max()

    # Fonction pour formater les axes en pourcentage
    def percent_format(x, _):
        return f'{x * 100:.1f}%'

    g4b, ax = plt.subplots()

    sns.scatterplot(
        data=df4b, 
        x=df4b['ctc_rate'], 
        y=df4b['rebound_rate'], 
        size=df4b['redirect_count'].rename('Redirect count'),
        sizes=(1000*(minsize/maxsize), 3000), # Je veux que l'écart de taille soit au moins du simple au triple pour bien voir
        color='cadetblue',
        legend='brief',
        ax=ax
    )

    ax.set(xlabel=f"\n Ratio of CTC per page visit", ylabel=f"Rebound rate\n")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_format))
    ax.xaxis.set_major_formatter(FuncFormatter(percent_format))

    # Déplacer la légende à droite
    plt.legend(
        title='Redirect count\n',
        title_fontsize=12,
        bbox_to_anchor=(1.05, 1), 
        loc=2, 
        borderaxespad=0,
        markerscale=1, 
        fontsize=10, 
        frameon=False,
        handlelength=5
    )

    # Ajouter des étiquettes sur le graph avec 'campaign_id'
    for i in range(df4b.shape[0]):
        plt.text(
            x=df4b['ctc_rate'].iloc[i],
            y=df4b['rebound_rate'].iloc[i],
            s=f"Page {df4b['page'].iloc[i]}",
            fontsize=9,
            fontstyle='italic',
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0, edgecolor='none')  # Fond blanc semi-transparent
        )

    st.header(f"Performance by page")
    st.subheader(f"Campaign : _{campaign}_")
    with st.expander("See comments"):
        st.write('''
            **Bubble size**: For each campaign, we want to check whether users were equally *redirected* to each of the 6 webpages.\n
            **Y axis**: *Rebound rates* inform us about the relevance of a campaign's redirects. If rebound rates are high on most pages, the campaign's targetting should be revised.\n
            **X axis**: *Click-to-call (CTC) rates* measure the number of callback requests made on each page, against the total number of visits on that same page. It hints at the preferences of the users reached by the campaign.
        ''')
    st.pyplot(g4b);
    