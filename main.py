import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title='Machine Learning Prognostic Model for the Overall Survival of Prostate Cancer Patients with Lymph Node-positive', layout='centered')

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.markdown('### Machine Learning Prognostic Model for the Overall Survival of Prostate Cancer Patients with Lymph Node-positive')

with st.expander("**Instructions**"):
    st.markdown("""
    1. **Input Patient Information**:
       - Select the patient's age, race, marital status, clinical T stage, PSA level, Gleason score, number of positive lymph nodes, and treatment details.
    
    2. **Predict Survival**:
       - After filling in the required patient data, click the "Predict" button to calculate the survival probabilities.
    
    3. **View Results**:
       - The model will display survival probabilities for 36 months, 60 months, 96 months, and 119 months.
       - A survival probability curve will also be generated to visualize the patient's predicted survival over time.
    """)

with st.expander("**Disclaimer**"):
    st.markdown("""
    This app is designed for healthcare professionals to aid in making informed decisions based on predicted survival outcomes for prostate cancer patients with lymph node-positive status. However, the results generated by this app are predictions based on a pre-trained machine learning model and should not be used as a sole basis for clinical decision-making. The app does not replace professional medical judgment and should be used in conjunction with other clinical information and expert advice.
    
    By using this app, you acknowledge that the results are provided for informational purposes only and that the developers are not responsible for any clinical decisions made based on these predictions. Please consult with a healthcare provider for personalized medical advice.
    """)

st.write("---")

age = st.selectbox("**Age**", ['≤60', '61-69', '≥70'],index=2,     help="Select the patient's age group. The age categories are ≤60 years, 61-69 years, and ≥70 years.")
if age == '61-69':
    age = [1, 0, 0]
elif age == '≤60':
    age = [0, 1, 0]
else:
    age = [0, 0, 1]
race = st.selectbox('**Race**', ['White', 'Black', 'Other'], help="Select the patient's race. Options are White, Black, or Other.")
if race == 'Black':
    race = [1, 0, 0]
elif race == 'Other':
    race = [0, 1, 0]
else:
    race = [0, 0, 1]
marital = st.selectbox('**Marital Status**', ['Married', 'Unmarried'], help="Select the patient's marital status. Choose between 'Married' or 'Unmarried'.")
if marital == 'Married':
    marital = [1, 0]
else:
    marital = [0, 1]
clinical = st.selectbox('**Clinical T stage**', ['T1-T3a', 'T3b', 'T4'], index=1, help="Select the patient's clinical T stage. This refers to the extent of the tumor. Options are T1-T3a, T3b, or T4.")
if clinical == 'T1-T3a':
    clinical = [1, 0, 0]
elif clinical == 'T3b':
    clinical = [0, 1, 0]
else:
    clinical = [0, 0, 1]
psa = st.number_input('**PSA level**', min_value=0.1, max_value=98.0, step=0.1, value=20.0, help="Enter the patient's PSA level, which is used to assess the risk of prostate cancer. PSA (Prostate-Specific Antigen) levels are measured in ng/mL.")
gs = st.selectbox('**Gleason Score**',
                  ['≤7(3+4)', '7(4+3)', '8', '≥9'],
                  index=3,
                  help=(
                      "The Gleason Score is used to assess how aggressive prostate cancer is. Here's what each score means:\n\n"
                      "- **≤7(3+4)**: Low grade, less aggressive cancer (Grade 3 is most common, lower risk of spreading).\n"
                      "- **7(4+3)**: Intermediate grade cancer (Grade 4 is most common, moderate risk of spread).\n"
                      "- **8**: High grade cancer (Both patterns are Grade 4, more aggressive and likely to spread).\n"
                      "- **≥9**: Very high grade, aggressive cancer (Both patterns are Grade 5, high risk of rapid spread)."
                  ))

if gs == '7(4+3)':
    gs = [1, 0, 0, 0]
elif gs == '8':
    gs = [0, 1, 0, 0]
elif gs == '≤7(3+4)':
    gs = [0, 0, 1, 0]
else:
    gs = [0, 0, 0, 1]
nodes = st.selectbox('**Number of positive lymph nodes**', ['1', '2', '≥3', 'No nodes were examined'], help="Select the number of positive lymph nodes, which can help determine the stage of cancer. Options include '1', '2', '≥3', or 'No nodes were examined'.")
if nodes == '≥3':
    nodes = [1, 0, 0, 0]
elif nodes == '1':
    nodes = [0, 0, 1, 0]
elif nodes == '2':
    nodes = [0, 0, 0, 1]
else:
    nodes = [0, 1, 0, 0]
therapy = st.selectbox('**Radical prostatectomy**', ['Yes', 'No'], help="Select whether the patient has undergone radical prostatectomy, which is a surgery to remove the prostate. Choose between 'Yes' or 'No'.")
if therapy == 'No':
    therapy = [1, 0]
else:
    therapy = [0, 1]
radio = st.selectbox('**Radiotherapy**', ['Yes', 'No'], index=1, help="Select whether the patient has received radiotherapy. Choose 'Yes' if the patient has received radiotherapy or 'No' if they have not.")
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
    #st.balloons()
    st.pyplot(fig)
    
# Sidebar UI
st.sidebar.markdown("""
<h2 style='color: black; margin-bottom: 0; font-size: 1em;'>Overview</h2>
<a href="https://www.nature.com/articles/s41598-023-45804-x" target="_blank";"><strong>Paper</strong></a> 
<p>   </p>
<p>This app is designed to predict the overall survival of prostate cancer patients with lymph node-positive status. Healthcare professionals can input patient-specific variables such as age, race, marital status, clinical T stage, PSA level, Gleason score, number of positive lymph nodes, and treatment details. Upon submission, the app utilizes a pre-trained Gradient Boosting Survival Analysis (GBSA) model to calculate and display survival probabilities and a survival probability curve, aiding in personalized patient care and informed decision-making.</p>
""", unsafe_allow_html=True)

st.sidebar.divider()

# Display the logo + feedback
st.sidebar.image("HU_Logo.svg", use_container_width="auto")
st.sidebar.write("**We value your feedback!** Please share your comments in our [discussions](https://www.healthuniverse.com/apps/ML%20Model%3A%20Survival%20of%20Lymph%20Node%2B%20Prostate%20Cancer%20Patients/discussions).")
