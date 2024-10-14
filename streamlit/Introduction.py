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

sns.reset_defaults()

warnings.simplefilter('ignore')

st.image("streamlit/images/garanteo.png")

st.title('Project Garanteo')

st.write('''Welcome to this Streamlit. These pages summarize my work on project Garanteo.\n
    Garanteo is a fictional insurance company running 100% online.\n
    As such, they need input on :\n
    - Their users, leads, clients\n
    - Which leads should be prioritized by their sales team\n 
    - How well their marketing campaigns have done so far\n
         
Project Garanteo is the final assignment of my 340h Data Analyst training with Databird.''')