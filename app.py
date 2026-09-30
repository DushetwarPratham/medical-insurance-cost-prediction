import streamlit as st
import pandas as pd
import pickle

# Load Saved Model and Preprocessing Files

# Load trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load scaler
with open('scaler.pkl','rb')as file:
    scaler = pickle.load(file)

# Load feature columns
with open('columns.pkl', 'rb') as file:
    columns = pickle.load(file)

# Application Title

st.title("Medical Insurance Cost Prediction")

st.write("Enter the customer's information below accurately to predict the estimated medical insurance charges.")

# User Inputs

age=st.number_input("Age", min_value=1,max_value=100,value=30)

sex=st.selectbox("Sex", ['female','male'])

bmi=st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

children = st.number_input("Number of Children", min_value=0, max_value=10,value=0)

smoker = st.selectbox("Smoker",['no','yes'])

region = st.selectbox("Region",['northeast','northwest','southeast','southwest'])


# Prediction

if st.button("Predict Insurance Cost"):
    
    # Create Input DataFrame
    input_data = pd.DataFrame({
        'age':[age],
        'sex':[sex],
        'bmi':[bmi],
        'children':[children],
        'smoker':[smoker],
        'region':[region]
    })

    # Show Input_Data
    st.write('Input Data:')
    st.dataframe(input_data)

    # One_hot Encoding

    input_data=pd.get_dummies(
        input_data,
        columns=['sex','smoker','region'],
        drop_first=True
    )

    # Match training columns

    input_data = input_data.reindex(columns=columns,fill_value=0)

    # Scaling

    numerical_features = ['age','bmi','children']
    input_data[numerical_features]=scaler.transform(input_data[numerical_features])

    # Prediction

    prediction = model.predict(input_data)

    # Display Results

    st.success(f"Estimated Insurance Cost: ₹{prediction[0]:,.2f}")
    