import tensorflow as tf
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder


# Load the Trained Model

model = load_model('model.h5')
# model = load_model('model_tuned.h5')

# Load the encoders and scaler

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('Scaler.pkl','rb') as file:
    Scaler=pickle.load(file)

# Streamlit Title
st.title("Customer Churn Prediction")


# Users input

geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,80)
balance = st.number_input('Balance')
creditScore = st.number_input('Credit Score')
estimatedSalary = st.number_input('Estimated Salary')
tenure = st.slider('Customer Tenure',0,10)
numOfProducts = st.slider('Number of Product',1,4)
hasCrCard = st.selectbox('Has Credit Card',[0,1])
isActiveMember = st.selectbox('Is Active Member',[0,1])



input_data = pd.DataFrame({

    'CreditScore' : [creditScore],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [numOfProducts],
    'HasCrCard' : [hasCrCard],
    'IsActiveMember' : [isActiveMember],
    'EstimatedSalary' : [estimatedSalary]

})

# One Hot encode Geography

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

 
# concatination of One Hot ecoded data (combine one hot encoded column with input data)

input_data =  pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Scale the input data

input_data_scaled = Scaler.transform(input_data)

# Predict Churn

prediction = model.predict(input_data_scaled)
Perdiction_prob = prediction[0][0]

st.write(f"Churn Probability: {Perdiction_prob: .2f}")

if Perdiction_prob >= 0.5:
    st.write('The Customar is likely to leave tha bank')
else:
    st.write('The Customar is not likely to leave tha bank')



