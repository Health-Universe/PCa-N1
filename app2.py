import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import matplotlib.pyplot as plt

st.set_page_config(page_title='ML', layout='wide')

st.title('Predict Survival Probability with Machine Learning ')
st.sidebar.subheader('Variable Select')
age = st.sidebar.selectbox('Age', ['≤60', '61-69', '≥70'])
if age == '61-69':
    age = [1, 0, 0]
elif age == '<=60':
    age = [0, 1, 0]
else:
    age = [0, 0, 1]
race = st.sidebar.selectbox('Race', ['White', 'Black', 'Other'])
if race == 'Black':
    race = [1, 0, 0]
elif race == 'Other':
    race = [0, 1, 0]
else:
    race = [0, 0, 1]
marital = st.sidebar.selectbox('Marital Status', ['Married', 'Unmarried'])
if marital == 'Married':
    marital = [1, 0]
else:
    marital = [0, 1]
clinical = st.sidebar.selectbox('Clinical T stage', ['T1', 'T2', 'T3', 'T4'])
if clinical == 'T1':
    clinical = [1, 0, 0, 0]
elif clinical == 'T2':
    clinical = [0, 1, 0, 0]
elif clinical == 'T3':
    clinical = [0, 0, 1, 0]
else:
    clinical = [0, 0, 0, 1]
psa = st.sidebar.number_input('PSA level', min_value=0.1, max_value=98.0, step=0.1)
gs = st.sidebar.selectbox('Gleason Score', ['≤6', '7(3+4)', '7(4+3)', '8', '≥9'])
if gs == '7(3+4)':
    gs = [1, 0, 0, 0, 0]
elif gs == '7(4+3)':
    gs = [0, 1, 0, 0, 0]
elif gs == '8':
    gs = [0, 0, 1, 0, 0]
elif gs == '≤6':
    gs = [0, 0, 0, 1, 0]
else:
    gs = [0, 0, 0, 0, 1]
nodes = st.sidebar.selectbox('Number of positive lymph nodes', ['1', '2', '3', '>3','None'])
if nodes == '>3':
    nodes = [1, 0, 0, 0, 0]
elif nodes == '1':
    nodes = [0, 0, 1, 0, 0]
elif nodes == '2':
    nodes = [0, 0, 0, 0, 1]
elif nodes == '3':
    nodes = [0, 0, 0, 1, 0]
else:
    nodes = [0, 1, 0, 0, 0]
radio = st.sidebar.selectbox('Radiotherapy', ['Yes', 'No'])
if radio == 'No':
    radio = [1, 0]
else:
    radio = [0, 1]
