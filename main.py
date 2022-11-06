import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import matplotlib.pyplot as plt

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

clinical_data = pd.read_csv('seer_prostate_data_N3.0.csv')
clinical_data = clinical_data.drop(['Unnamed: 0'], axis=1)
survival_time = clinical_data['Time']
survival_status = clinical_data['Event']
merge_df = clinical_data.drop(['Time', 'Event'], axis=1)

survival_target = pd.concat([survival_time, survival_status], axis=1)
survival_target['Event'] = survival_target['Event'].map({'Alive': False, 'Dead': True})

x_train_n, x_test_n, y_train, y_test = train_test_split(merge_df, survival_target, test_size=0.3, random_state=123)

def onehot(_x_train_n):
    data_cata_train = _x_train_n.iloc[:, 0:7]
    data_int_train = _x_train_n.iloc[:, 7:]
    encoder = OneHotEncoder()
    x_train_cata = encoder.fit_transform(data_cata_train).toarray()
    x_train_cata = pd.DataFrame(x_train_cata)
    x_train_cata.columns = encoder.get_feature_names_out()
    data_int_train = data_int_train.astype('float')
    data_int_train = data_int_train.reset_index(drop=True)
    _x_train = pd.concat([x_train_cata, data_int_train], axis=1)
    return _x_train

x_train = onehot(x_train_n)
new_y_train = [(y_train['Event'].iloc[i], y_train['Time'].iloc[i]) for i in range(y_train.shape[0])]
new_y_train = np.array(new_y_train, dtype=[('status', 'bool'), ('time', '<f8')])

RSFsurvival = RandomSurvivalForest(n_estimators=95,
                                   max_depth=5,
                                   min_samples_leaf=3,
                                   min_samples_split=15,
                                   n_jobs=8,
                                   random_state=0)
RSFsurvival.fit(x_train, new_y_train)

GDBT = GradientBoostingSurvivalAnalysis(learning_rate=0.4,
                                        n_estimators=50,
                                        min_samples_leaf=6,
                                        min_samples_split=2,
                                        max_depth=3,
                                        subsample=1,
                                        dropout_rate=0,
                                        random_state=0)
GDBT.fit(x_train, new_y_train)

model = st.selectbox('Model Select', ['RSF', 'GDBT'])
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
    if model == 'RSF':
        prob = RSFsurvival.predict(x_test)
        st.subheader('RSF model score: '+ str(round(prob[0],2)))
        st.subheader('Survival probability between 12 and 60 months:')
        surv = RSFsurvival.predict_survival_function(x_test, return_array=True)
        yv = []
        for j, s in enumerate(surv):
            for j in range(0, len(s)):
                if RSFsurvival.event_times_[j] in (12, 24, 36, 48, 60):
                    yv.append(s[j])
                    #st.metric('X:'+str(RSFsurvival.event_times_[j]), s[j])
            plt.step(RSFsurvival.event_times_[12:61], s[12:61], where="post")
        y_df = pd.DataFrame(yv).T
        y_df.columns = ['12months', '24months', '36months', '48months', '60months']
        st.table(y_df)
        plt.ylabel("Survival probability")
        plt.xlabel("Time in months")
        plt.title('Months to survival probability')
        plt.legend()
        plt.grid(True)
        st.balloons()
        st.pyplot(fig)
    else:
        prob = GDBT.predict(x_test)
        st.subheader('GDBT model score: ' + str(round(prob[0], 2)))
        st.subheader('Survival Probability between 12 and 60 days:')
        surv = GDBT.predict_survival_function(x_test)
        yv = []
        for fn in surv:
            for i in range(0, len(fn.x)):
                if fn.x[i] in (12, 24, 36, 48, 60):
                    yv.append(fn(fn.x)[i])
                    # st.metric('X:' + str(fn.x[i]), fn(fn.x)[i])
            plt.step(fn.x[12:61], fn(fn.x)[12:61], where="post")
        y_df = pd.DataFrame(yv).T
        y_df.columns = ['12days', '24days', '36days', '48days', '60days']
        st.table(y_df)
        plt.ylabel("Survival probability")
        plt.xlabel("Time in months")
        plt.title('Months to Survival Probability')
        plt.legend()
        plt.grid(True)
        st.balloons()
        st.pyplot(fig)

