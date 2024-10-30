import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import os
import base64

# Page configuration
st.set_page_config(layout="wide", page_title="Car Dheko - Price Prediction", page_icon="🚗")

# Load paths
logo_path = "E:/UDHAYA/Cardeko_project/Cardeko_logo.png"
dataset_path = "E:/UDHAYA/Cardeko_project/Processed_dataset.csv"
model_path = "E:/UDHAYA/Cardeko_project/pipeline_model.pkl"

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #FFFFFF; 
        text-align: center;
    }
    .description {
        text-align: center;
        color: #FFFFFF;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #ff6600;
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 10px;
    }
    .sidebar-content {
        font-size: 18px;
        color: #ff6600;
    }
    </style>
""", unsafe_allow_html=True)

# Load logo
if os.path.exists(logo_path):
    try:
        # Open the logo and resize
        with open(logo_path, "rb") as image_file:
            logo_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Display the logo in the center with specific width
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{logo_base64}" width="200"/>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Error loading logo: {e}")
else:
    st.warning("Logo file not found.")

# App title and description
st.markdown('<p class="title">✨Car Dheko - Used Car Price Prediction✨</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Get an estimated price for your car based on specifications and history.</p>', unsafe_allow_html=True)

# Load data and model functions
def load_data():
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    else:
        st.error("Dataset not found.")
        return None

def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model not found.")
        return None

# Initialize data and model
df = load_data()
pipeline_model = load_model()

if df is not None and pipeline_model is not None:
    st.sidebar.markdown("<p class='sidebar-content'>**Car Specifications**</p>", unsafe_allow_html=True)
    
    # Sidebar inputs for user specifications
    brand = st.sidebar.selectbox("Car Brand", options=df['Brand'].unique())
    fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Lpg', 'Cng', 'Electric'])
    body_type = st.sidebar.selectbox("Body Type", ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans', 'Convertibles', 'Hybrids', 'Wagon', 'Pickup Trucks'])

    # Model dropdown based on filters
    filtered_models = df[(df['Brand'] == brand) & (df['body type'] == body_type) & (df['Fuel type'] == fuel_type)]['model'].unique()
    if filtered_models.size > 0:
        car_model = st.sidebar.selectbox("Car Model", options=filtered_models)
    else:
        car_model = st.sidebar.selectbox("Car Model", options=["No models available"])

    transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
    seats = st.sidebar.selectbox("Seats", sorted(df['Seats'].unique()))
    insurance_type = st.sidebar.selectbox("Insurance Type", ['Third Party insurance', 'Comprehensive', 'Third Party', 'Zero Dep', '2', '1', 'Not Available'])
    color = st.sidebar.selectbox("Color", df['Color'].unique())
    city = st.sidebar.selectbox("City", options=df['City'].unique())

    # Numeric inputs
    model_year = st.sidebar.number_input("Manufacturing Year", min_value=1980, max_value=2025, step=1)
    mileage = st.sidebar.number_input("Mileage (in km/l)", min_value=1.0, max_value=50.0,step=0.1)
    owner_no = st.sidebar.number_input("Owner Number", min_value=1, max_value=5, step=1)
    kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=100, max_value=1000000, step=1000)

    # Main area predict button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🚗 Predict Car Price")

    # Predict
    if predict_button:
        if car_model != "No models available":
            input_data = pd.DataFrame({
                'Fuel type': [fuel_type],
                'body type': [body_type],
                'transmission': [transmission],
                'ownerNo': [owner_no],
                'Brand': [brand],
                "model": [car_model],
                'modelYear': [model_year],
                'Insurance Type': [insurance_type],
                'Kms Driven': [kms_driven],
                'Mileage': [mileage],
                'Seats': [seats],
                'Color': [color],
                'City': [city]
            })

            try:
                prediction = pipeline_model.predict(input_data)
                st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")
            except Exception as e:
                st.error("Error making prediction.")
        else:
            st.warning("Please select valid options for all fields.")
else:
    st.error("Unable to load data or model.")
