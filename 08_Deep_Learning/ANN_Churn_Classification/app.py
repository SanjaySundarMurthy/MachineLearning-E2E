import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load the Models
model=load_model('final_model.h5')

# Load the Encoders

with open('label_encode_gender.pkl','rb') as f:
    label_encoder_gender=pickle.load(f)

with open('onehotencode.pkl','rb') as f:
    onehotencoder=pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)

## Title
st.title("Customer Churn prediction")

# Initialize the variables
credit_score= st.number_input('Credit Score')
age=st.slider('Age',18,90)
tenure= st.slider('Tenure',0,10)
balance=st.number_input('Balance')
num_of_products= st.slider('Number of Products',1,4)
has_cr_card= st.selectbox('Has Credit Card', [0,1])
isactivemember= st.selectbox('Is Active Member', [0,1])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
geography=st.selectbox('geography', onehotencoder.categories_[0])
estimate_sal=st.number_input('EstimatedSalary')

## Prepare the input data

input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [isactivemember],
    'EstimatedSalary': [estimate_sal],
})

geo_encoded = onehotencoder.transform([[geography]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded,columns=onehotencoder.get_feature_names_out())

## combine the data with input data

input_data= pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


# scale the data

input_data_scaled= scaler.transform(input_data)

## Prediction

predict_churn= model.predict(input_data_scaled)

prediction_proba=predict_churn[0][0]
st.write(prediction_proba)

if prediction_proba > 0.5:
    st.write("Customer likely to churn")
else:
    st.write("Customer not likely to churn")