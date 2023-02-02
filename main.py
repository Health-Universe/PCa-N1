import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title='ML', layout='centered')

st.title('Machine Learning Prognostic Model for the Overall Survival of Prostate Cancer Patients with Lymph Node-positive ')
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
clinical = st.sidebar.selectbox('Clinical T stage', ['T1-T3a', 'T3b', 'T4'])
if clinical == 'T1-T3a':
    clinical = [1, 0, 0]
elif clinical == 'T3b':
    clinical = [0, 1, 0]
else:
    clinical = [0, 0, 1]
psa = st.sidebar.number_input('PSA level', min_value=0.1, max_value=98.0, step=0.1, value=20.0)
gs = st.sidebar.selectbox('Gleason Score', ['≤7(3+4)', '7(4+3)', '8', '≥9'])
if gs == '7(4+3)':
    gs = [1, 0, 0, 0]
elif gs == '8':
    gs = [0, 1, 0, 0]
elif gs == '≤7(3+4)':
    gs = [0, 0, 1, 0]
else:
    gs = [0, 0, 0, 1]
nodes = st.sidebar.selectbox('Number of positive lymph nodes', ['1', '2', '≥3', 'No nodes were examined'])
if nodes == '≥3':
    nodes = [1, 0, 0, 0]
elif nodes == '1':
    nodes = [0, 0, 1, 0]
elif nodes == '2':
    nodes = [0, 0, 0, 1]
else:
    nodes = [0, 1, 0, 0]
therapy = st.sidebar.selectbox('Radical prostatectomy', ['Yes', 'No'])
if therapy == 'No':
    therapy = [1, 0]
else:
    therapy = [0, 1]
radio = st.sidebar.selectbox('Radiotherapy', ['Yes', 'No'])
if radio == 'No':
    radio = [1, 0]
else:
    radio = [0, 1]

GBSA = joblib.load('GBSA12.18.pkl')
query = st.button('Predict')
if query:
    age.extend(race)
    age.extend(marital)
    age.extend(clinical)
    age.extend(radio)
    age.extend(therapy)
    age.extend(nodes)
    age.extend(gs)
    data_list = age
    x_df = pd.DataFrame(data_list).T
    x_df.columns =['Age_61-69','Age_<=60','Age_>=70','Race_Black','Race_Other','Race_White','Marital_Married','Marital_Unmarried','CS.extension_T1_T3a','CS.extension_T3b','CS.extension_T4','Radiation_None/Unknown','Radiation_Yes','Therapy_None','Therapy_RP','Nodes.positive_>=3','Nodes.positive_None','Nodes.positive_One','Nodes.positive_Two','Gleason.Patterns_4+3','Gleason.Patterns_8','Gleason.Patterns_<=3+4','Gleason.Patterns_>=9']
    x_psa_df = pd.DataFrame([psa],columns=['PSA'])
    x_test = pd.concat([x_df, x_psa_df], axis=1)
    fig = plt.figure()
    prob = GBSA.predict(x_test)
    #st.subheader('GBSA model score: ' + str(round(prob[0], 2)))
    #st.subheader('Survival Probability:')
    surv = GBSA.predict_survival_function(x_test)
    yv = []
    for fn in surv:
        for i in range(0, len(fn.x)):
            if fn.x[i] in (36, 60, 96, 119):
                yv.append(fn(fn.x)[i])
                    # st.metric('X:' + str(fn.x[i]), fn(fn.x)[i])
    #plt.step(fn.x[0:120], fn(fn.x)[0:120], where="post")
    plt.plot(fn.x[0:120], fn(fn.x)[0:120])
    y_df = pd.DataFrame(yv).T
    y_df.columns = ['36-month', '60-month', '96-month', '119-month']
    y_df.index = ['Survival probability']
    st.table(y_df)
    plt.xlim(0,120)
    plt.ylabel("Survival probability")
    plt.xlabel("Survival time (mo)")
    #plt.title('Survival Probability')
    plt.legend()
    plt.grid(True)
    st.balloons()
    st.pyplot(fig)
