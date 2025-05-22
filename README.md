# ANN_JNS
My First ANN model 

# 🧠 Customer Churn Prediction using ANN

This project implements an **Artificial Neural Network (ANN)** to predict customer churn using the `Churn_Modelling.csv` dataset. The goal is to determine whether a customer will leave the bank or not based on their demographic and account-related features.

## 📂 Dataset Information

The dataset contains the following features:

- **CreditScore** – Customer's credit score  
- **Geography** – Country of residence (France, Spain, Germany)  
- **Gender** – Male or Female  
- **Age** – Age of the customer  
- **Tenure** – Number of years the customer has been with the bank  
- **Balance** – Bank account balance  
- **NumOfProducts** – Number of bank products the customer uses  
- **HasCrCard** – Does the customer have a credit card (1 = Yes, 0 = No)  
- **IsActiveMember** – Activity status of the customer (1 = Active, 0 = Inactive)  
- **EstimatedSalary** – Customer’s estimated salary  
- **Exited** – Target variable (1 = Customer left the bank, 0 = Customer stayed)

## 🧪 Technologies Used

- **Python**
- **Pandas & NumPy**
- **Scikit-learn** for preprocessing and splitting
- **TensorFlow / Keras** for building the ANN
- **TensorBoard** for visualizing training progress

## 🏗️ Model Architecture

The ANN consists of:
- Input Layer
- Two Hidden Layers with ReLU activation
- Output Layer with Sigmoid activation for binary classification

## ⚙️ Preprocessing Steps

1. Removed irrelevant columns like `RowNumber`, `CustomerId`, and `Surname`.
2. Encoded categorical columns:
   - Label Encoding for `Gender`
   - One-Hot Encoding for `Geography`
3. Standardized features using `StandardScaler`.

## 🎯 Objective

The aim is to build a predictive model that can assist banks in identifying customers who are likely to churn, enabling proactive customer retention strategies.

## 📝 Results

Training is done using TensorFlow, with:
- **EarlyStopping** to prevent overfitting
- **TensorBoard** for tracking metrics

## 📦 Output

- Trained model saved as `model.h5`
- Encoders and scaler saved as `.pkl` files

## 📁 Folder Structure

project/
│
├── model.h5
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── Scaler.pkl
├── log/fit/<timestamp>/ (TensorBoard logs)
└── ProANN.ipynb

🧪 How to Run Locally
0.1 Application Url
https://annjns-afap7supqqrsjoo2fswute.streamlit.app/
