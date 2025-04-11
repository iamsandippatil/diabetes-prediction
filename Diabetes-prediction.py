# import all libraries
import numpy as np
import pickle
import streamlit as st
import os

import sklearn 
print(sklearn.__version__)
print(os.path.abspath('trained_diabetes_model.sav'))  # Verify absolute path



# Loading the saved model
loaded_model = pickle.load(open(os.path.abspath('trained_diabetes_model.sav'), 'rb'))

# Creating a function for prediction

def diabetes_prediction(input_data):
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    # Changing the data into a NumPy array
    input_data_as_nparray = np.asarray(input_data)

    # Reshaping the data since there is only one instance
    input_data_reshaped = input_data_as_nparray.reshape(1, -1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data_reshaped)

    # Make prediction
    prediction = loaded_model.predict(input_data_scaled)

    if prediction == 0:
        return 'Non Diabetic'
    else:
        return 'Diabetic'

def main():

    # Giving a title
    st.title('Diabetes Prediction Web App')

    # Getting input from the user
    try:
        Pregnancies = float(st.text_input('No. of Pregnancies:', '0'))
        Glucose = float(st.text_input('Glucose level:', '0'))
        BloodPressure = float(st.text_input('Blood Pressure value:', '0'))
        SkinThickness = float(st.text_input('Skin thickness value:', '0'))
        Insulin = float(st.text_input('Insulin level:', '0'))
        BMI = float(st.text_input('BMI value:', '0'))
        DiabetesPedigreeFunction = float(st.text_input('Diabetes pedigree function value:', '0'))
        Age = float(st.text_input('Age:', '0'))
    except ValueError:
        st.error("Please enter valid numeric values.")

    # Code for prediction
    diagnosis = ''

    # Making a button for prediction
    if st.button('Predict'):
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        
        diagnosis = diabetes_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()
