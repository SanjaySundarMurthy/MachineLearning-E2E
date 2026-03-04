import streamlit as st
import tensorflow 
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

## load the models 
model= load_model('model.h5')

# load encoders
with open('onehotencode.pkl','rb') as f:
    onehotencoder=pickle.load(f)
with open('label_encode_gender.pkl','rb') as f:
    labelencoder=pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)

## 
## Title
st.title("Customer Salary prediction")

# Initialize the variables
credit_score= st.slider('Credit Score',0,100)
age=st.slider('Age',18,90)
tenure= st.slider('Tenure',0,10)
balance=st.slider('Balance',1000,999999)
num_of_products= st.slider('Number of Products',1,4)
has_cr_card= st.selectbox('Has Credit Card', [0,1])
isactivemember= st.selectbox('Is Active Member', [0,1])
gender=st.selectbox('Gender',labelencoder.classes_)
geography=st.selectbox('Geography',onehotencoder.categories_[0])
exited=st.selectbox('Exited',[0,1])
## Prepare the input data
input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [labelencoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [isactivemember],
    'Exited': [exited],
})

## Do not use fit_transform on Test DATA always use transform only 
one_encoder_geo=onehotencoder.transform([[geography]])
one_geo_df=pd.DataFrame(one_encoder_geo.toarray(),columns=onehotencoder.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),one_geo_df],axis=1)

input_scaled=scaler.transform(input_data)
prediction=model.predict(input_scaled)
st.write(f'The Salary of the Employee is : {prediction[0][0]:.2f}')