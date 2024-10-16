## Initialisation

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import datetime as dt
import warnings
import streamlit as st

warnings.simplefilter('ignore')

# Chargement des fichiers
df_campaigns_metrics = pd.read_csv(r"streamlit/output_streamlit/campaigns_metrics.csv", index_col=0)

dict_choices = {
    'Click':{
        'graph1_y_bars':'sum_click',
        'graph1_y_strips':'cost_per_click',
        'graph1_yleft_label':'Click count',
        'graph1_yright_label':'Cost per click',
        'graph2_x':'mean_click',
        'graph2_xlabel':'Clicks per session',
        'caption1':"We note that the costs-per-click are quite close, with a low standard deviation / mean ratio. In fact the performance gaps are more tangible with the cost-per-prospect graph.",
        'caption2':"On the horizontal axis, the graph computes the average number of clicks made during a session initiated by each campaign. "
    },
    'Prospect':{
        'graph1_y_bars':'prospect_created',
        'graph1_y_strips':'cost_per_prospect',
        'graph1_yleft_label':'Prospects created',
        'graph1_yright_label':'Cost per prospect created',
        'graph2_x':'session_to_prospect_rate',
        'graph2_xlabel':'Session to prospect : conversion rate',
        'caption1':"Since they have a higher standard deviation / mean ratio, the costs-per-prospect are the most relevant metrics to assess the relative campaigns' ROIs (in absence of revenue or CLTV data).",
        'caption2':"Session-to-prospect ratio measures the number of prospects created thanks to each campaign, divided by the total number of sessions initiated by the same campaign. It is the most relevant metric to measure the overall efficiency of a campaign."
    },
    'Client':{
        'graph1_y_bars':'client_acquired',
        'graph1_y_strips':'cost_per_client',
        'graph1_yleft_label':'Clients acquired',
        'graph1_yright_label':'Cost per client acquisition',
        'graph2_x':'session_to_client_rate',
        'graph2_xlabel':'Session to client : conversion rate',
        'caption1':"This KPI has a limitation: unlike prospect creation, client acquisition does not only rely on marketing campaigns, but also on the performance of Garanteo's sales team, for which we do not have any cost data.",
        'caption2':"Session-to-prospect ratio measures the final number of clients acquired thanks to each campaign, divided by the total number of sessions initiated by the same campaign. This KPI has a limitation: unlike prospect creation, client acquisition does not only rely on marketing campaigns, but also on the performance of Garanteo's sales team."
    },
}

# Sélection de la base

focus_options = ['Click', 'Prospect', 'Client']
campaign_focus = st.sidebar.selectbox("Select a basis for KPIs calculations :", focus_options)
dict_graphs = dict_choices[campaign_focus]

st.title('Garanteo | Overview of Campaigns')

tab_a, tab_b = st.tabs([
    f":coin: Cost per {campaign_focus}", 
    ":dart: Targetting"
])

with tab_a:

    # Graph Coût total & CPC / CPPro / CPClient

    x = df_campaigns_metrics['campaign_id']
    g3a, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    sns.barplot(
        df_campaigns_metrics, 
        x=x, 
        y=df_campaigns_metrics[dict_graphs['graph1_y_bars']], 
        palette = 'mako', 
        hue=df_campaigns_metrics['campaign_type'].rename('Campaign type').str.capitalize(), 
        ax=ax1
    )
    sns.stripplot(
        df_campaigns_metrics, 
        x=x, 
        y=df_campaigns_metrics[dict_graphs['graph1_y_strips']], 
        color='orange', 
        ax=ax2
    )

    ax1.set_xlabel(f"\nCampaign id")
    ax1.set_ylabel(f"{dict_graphs['graph1_yleft_label']}\n")
    ax2.set_ylabel(f"\n{dict_graphs['graph1_yright_label']}")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=70)

    # On calcule la moyenne du coût par clic (cpc)
    mean_cost = round(df_campaigns_metrics['total_cost'].sum() / df_campaigns_metrics[dict_graphs['graph1_y_bars']].sum(), 2)

    # Ajout d'une ligne horizontale en pointillés au niveau de la moyenne
    ax2.axhline(mean_cost, color='orange', linestyle='--', label=f'Mean CPC: {mean_cost}')

    # On calcule l'écart-type du cpc (on revient à la formule mathématique pour bien pondérer par le nb de clics) 
    std_cost = round(np.sqrt(
                        1/df_campaigns_metrics[dict_graphs['graph1_y_bars']].sum()\
                        *(
                            df_campaigns_metrics[dict_graphs['graph1_y_bars']]\
                            *(df_campaigns_metrics[dict_graphs['graph1_y_strips']] - mean_cost)**2
                        ).sum()
                    ),2)
    std_over_mean_cost = '%.1f'%(std_cost/mean_cost*100)

    st.header(f"Comparison of {dict_graphs['graph1_yright_label'].lower()}")
    col1, col2 = st.columns(2)
    col1.metric(":orange[---] Mean", f"{mean_cost}€ per {campaign_focus.lower()}")
    col2.metric("Standard deviation", f"{std_over_mean_cost}% of mean")
    with st.expander("See comments"):
        st.write(f'''
            Here we want to compare the return on investment for each campaign:\n
            - The vertical bars show the {dict_graphs['graph1_yleft_label'].lower()}\n
            - The dots show the {dict_graphs['graph1_yright_label'].lower()}\n
            {dict_graphs['caption1']}
        ''')
    st.pyplot(g3a)
    

with tab_b:
    # Taux de rebond vs taux de conversion

    df3 = df_campaigns_metrics

    minsize = df3['total_cost'].min()
    maxsize = df3['total_cost'].max()

    # Fonction pour formater les axes en pourcentage
    def percent_format(x, _):
        return f'{x * 100:.1f}%'

    # Graph des taux de rebond vs taux de conversion

    g3b, ax = plt.subplots()

    sns.scatterplot(
        data=df3, 
        x=df3[dict_graphs['graph2_x']], 
        y=df3['rebound_rate'].rename('Rebound rate\n'), 
        size=df3['total_cost'].rename('\nTotal cost in EUR'),
        sizes=(minsize/10, maxsize/10),
        hue=df3['campaign_type'].rename('Campaign type').str.capitalize(),
        palette='mako',
        legend='brief',
        ax=ax
    )

    ax.set(xlabel=f"\n {dict_graphs['graph2_xlabel']}")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_format))

    if campaign_focus != "Click":
        ax.xaxis.set_major_formatter(FuncFormatter(percent_format))

    # Déplacer la légende à droite
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc=2, 
        borderaxespad=0.,
        markerscale=0.8, 
        fontsize=10, 
        frameon=False
    )

    # Ajouter des étiquettes sur le graph avec 'campaign_id'
    for i in range(df3.shape[0]):
        plt.text(
            x=df3[dict_graphs['graph2_x']].iloc[i],
            y=df3['rebound_rate'].iloc[i],
            s=df3['campaign_id'].iloc[i],
            fontsize=9,
            fontstyle='italic',
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0, edgecolor='none')  # Fond blanc semi-transparent
        )

    st.header(f"Comparison of campaigns' targetting")
    st.subheader(f"Best targetting in the bottom right corner")
    with st.expander("See comments"):
        st.write(f'''
            For all campaigns, we want to check whether users were targetted in an appropriate way.\n
            Rebound rates inform us about the relevance of a campaign's redirects. If rebound rates are high on most pages, the campaign's targetting should be revised.\n
            {dict_graphs['caption2']}
        ''')
    st.pyplot(g3b);