import warnings
import streamlit as st

warnings.simplefilter('ignore')

st.image("streamlit/images/garanteo.png")

st.title('Project Garanteo')

st.markdown("**Welcome to this Streamlit. These pages summarize my work on project Garanteo.**")
with st.container(border=True):
    st.write('''
        *Garanteo is a fictional insurance company running 100% online.*\n
        *As such, they need input on :*\n
        - *Their users, leads, clients*\n
        - *Which leads should be prioritized by their sales team*\n 
        - *How well their marketing campaigns have done so far*\n
    ''')         
st.write("**Project Garanteo is the final assignment of my 340h Data Analyst training with Databird.**")