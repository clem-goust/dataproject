# Initialisation

import warnings
import streamlit as st

warnings.simplefilter('ignore')

# Page de conclusion

st.title("Garanteo | Recommendations")

with st.container(border=False):
    st.subheader(":star: Prioritizing high-potential leads")
    with st.expander("See details"):
        st.write('''
             - Thanks to a Machine learning model, we predicted probabilities for leads to convert into clients after a call from the Sales team.
             - We converted these probabilities into scores and built a Leads dashboard for the Sales team.
             - Garanteo's teams should prioritize highly-rated leads and among these, leads with older callback demands.  
             ''')
    
with st.container(border=False):
    st.subheader(":money_with_wings: Reallocating budget from outlying campaigns")
    with st.expander("See details"):
        st.write('''
             - Plotting the Costs-per-prospect helped us identify the less efficient campaigns (at least 3 outliers).
             - Coupled with high rebound rated, these are hints of targetting and redirecting issues. Our web trafic graphs provide more insight.
             - We would recommend stopping the 3 or 4 outlying campaigns and investing new marketing channels, such as social media, insurance comparators, sponsored articles etc.   
             ''')

with st.container(border=False):
    st.subheader(":dart: Using our clustering data to inform ads targeting")
    with st.expander("See details"):
        st.write('''
             - We also built clusters on users' preferences and connection behaviors (see distribution in Users overview) 
             - This data may be used to inform ads targeting (content of the ads, appropriate time to activate ads)   
             ''')