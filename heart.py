import streamlit as st
import pandas as pd
import pickle

st.write("""
# Simple Heart Disease Prediction

This app predicts the **Heart disease** !
""")

st.header('User Input Parameters')

def user_input_features():
    Age = st.slider('Age', 20.0,80.0,60.0)
    Cigretes_Per_Day = st.slider('Cigretes Per Day', 0.0,70.0,15.0)
    Total_Cholestrol = st.slider('Total Cholestrol', 110.0, 670.0,254.0)
    Sys_BP = st.slider('Sys BP', 110.0,300.0,177.0)
    Glucose = st.slider('Glucose',30.0,400.0,79.0)
    data = {'Age': Age,
            'Cigretes Per Day': Cigretes_Per_Day,
            'Total Cholestrol': Total_Cholestrol,
            'Sys BP': Sys_BP,
            'Glucose':Glucose}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)
filename = 'finalized_model.pkl'
clf= pickle.load(open(filename, 'rb'))

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

#st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)
mapping={0:'No',1:'Yes'}
st.subheader('Prediction')
st.write(mapping[prediction[0]])



st.subheader('Prediction Probability')
st.write(prediction_proba)
