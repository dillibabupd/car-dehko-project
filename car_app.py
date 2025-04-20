import streamlit as st
import pandas as pd
import pickle as pk

# Load trained model
model = pk.load(open("D:\guvi\code\car_model.pkl", 'rb'))

# Load dataset
car_data = pd.read_csv('car_data.csv') 

st.header('🚗 Car Price Prediction ML Model')

c1, c2 = st.columns(2)
with c1:
    # Select brand
    brand = st.selectbox('Select Car Brand', sorted(car_data['manufacturer'].unique()))
    # Dynamically filter model options based on selected brand
    filtered_models = car_data[car_data['manufacturer'] == brand]['model'].unique()
    car_model = st.selectbox('Select Car Model', sorted(filtered_models))
    filtered_body = car_data[car_data['manufacturer'] == brand]['body_type'].unique()
    car_body = st.selectbox('Body type',sorted(filtered_body))
    filtered_seats = car_data[car_data['manufacturer'] == brand]['Seats'].unique()
    seats = st.selectbox('seats',sorted(filtered_seats))
    # Other inputs

    fuel = st.selectbox('Fuel Type', (car_data['Fuel_Type'].unique()))
    insurance = st.selectbox('Insurance Validity', (car_data['Insurance Validity'].unique()))
    transmission = st.selectbox('Transmission Type', (car_data['transmission_type'].unique()))
    owner = st.selectbox('No. of Owners', (car_data['owner_No'].unique()))
with c2: 
    kms = st.slider('No of kms driven', 10 , 200000) 
    year = st.slider('Manufacture Year', 1990, 2024)
    milage = st.slider('Mileage (kmpl)', 5, 30)
    engine = st.slider('Engine (CC)', 700, 1500)
    power = st.slider('Max Power (BHP)', 0, 200)
    torque = st.slider('Torque (Nm)', 100, 3000)
    

# Predict button
if st.button('Predict'):
    input_data = pd.DataFrame(
        [[fuel, car_body, kms, transmission, owner, brand, car_model, year, insurance, milage, engine, power, torque, seats]],
        columns=[
            'Fuel_Type', 'body_type', 'kms_driven', 'transmission_type', 'owner_No',
            'manufacturer', 'model', 'Manufacture_year', 'Insurance Validity',
            'Mileage', 'Engine', 'Max Power', 'Torque', 'Seats'
        ]
    )

    # Load label encoders
    le_fuel = pk.load(open("D:/guvi/code/le_fuel.pkl", 'rb'))
    le_body = pk.load(open("D:/guvi/code/le_body.pkl", 'rb'))
    le_trans = pk.load(open("D:/guvi/code/le_transmission.pkl", 'rb'))
    le_model = pk.load(open("D:/guvi/code/le_model.pkl", 'rb'))
    le_man = pk.load(open("D:/guvi/code/le_manufacturer.pkl", 'rb'))
    le_ins = pk.load(open("D:/guvi/code/le_insurance.pkl", 'rb'))

    # Transform input values using the saved encoders
    input_data['Fuel_Type'] = le_fuel.transform(input_data['Fuel_Type'])
    input_data['body_type'] = le_body.transform(input_data['body_type'])
    input_data['transmission_type'] = le_trans.transform(input_data['transmission_type'])
    input_data['manufacturer'] = le_man.transform(input_data['manufacturer'])
    input_data['model'] = le_model.transform(input_data['model'])
    input_data['Insurance Validity'] = le_ins.transform(input_data['Insurance Validity'])

    st.subheader("🚙 Processed Input")
    st.write(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    st.markdown(f"### 💰 Estimated Car Price: ₹ {prediction[0]:,.2f}")
