import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title='ML', layout='centered')

st.title('Predict Survival Probability with Machine Learning ')
st.sidebar.subheader('Variable Select')
age = st.sidebar.selectbox('Age', ['≤60', '61-69', '≥70'])
if age == '61-69':
    age = [1, 0, 0]
elif age == '≤60':
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

RSF = joblib.load('GBSA.pkl')
query = st.button('Predict')
if query:
    age.extend(race)
    age.extend(marital)
    age.extend(clinical)
    age.extend(radio)
    age.extend(gs)
    age.extend(nodes)
    data_list = age
    x_df = pd.DataFrame(data_list).T
    x_df.columns =['Age_61-69','Age_<=60','Age_>=70','Race_Black','Race_Other','Race_White','Marital_Married','Marital_Unmarried','CS.extension_T1','CS.extension_T2','CS.extension_T3','CS.extension_T4','Radiation_None/Unknown','Radiation_Yes','Gleason.Patterns_3+4','Gleason.Patterns_4+3','Gleason.Patterns_8','Gleason.Patterns_<=6','Gleason.Patterns_>=9','Nodes.positive_>3','Nodes.positive_None','Nodes.positive_One','Nodes.positive_Three','Nodes.positive_Two']
    x_psa_df = pd.DataFrame([psa],columns=['PSA'])
    x_test = pd.concat([x_df, x_psa_df], axis=1)
    fig = plt.figure()
    prob = RSF.predict(x_test)
    st.subheader('RSF  score: '+ str(round(prob[0],2)))
    st.subheader('Survival probability between 12 and 60 months:')
    surv = RSF.predict_survival_function(x_test, return_array=True)
    yv = []
    for j, s in enumerate(surv):
        for j in range(0, len(s)):
            if RSF.event_times_[j] in (36, 60, 96, 115):
                    yv.append(s[j])
                    #st.metric('X:'+str(RSFsurvival.event_times_[j]), s[j])
    plt.step(RSF.event_times_[0:120], s[0:120], where="post")
    y_df = pd.DataFrame(yv).T
    y_df.columns = ['36months', '60months', '96months', '115months']
    st.table(y_df)
    plt.xlim(0,120)
    plt.ylabel("Survival probability")
    plt.xlabel("Time in months")
    plt.title('Survival Probability')
    plt.legend()
    plt.grid(True)
    st.balloons()
    st.pyplot(fig)
