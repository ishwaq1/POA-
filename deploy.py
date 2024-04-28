import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data import *
import pickle

# Set page config
st.set_page_config(page_title="Depot Demand Predictions", page_icon="🚚", layout="wide")

# Function to make predictions based on input features
def make_prediction(drivers, pickups_per_day, inventory_level, depot, quantity_delivered, discounts, demand):
    # This is just a placeholder for demonstration purposes
    predicted_demand = np.random.randint(100, 1000)  # Replace this with your actual prediction logic
    return predicted_demand

# Sidebar for navigation
with st.sidebar:
    from streamlit_option_menu import option_menu
    selected = option_menu(None, ["Home", "Prediction"], 
        icons=['house', 'bar-chart-line'], 
        menu_icon="cast", default_index=0, orientation="vertical",
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        })

if selected == "Home":
    st.title('Welcome to Depot Demand Predictions')
    st.markdown("""
        Use this tool to predict daily demand at different depots based on various input features. 
        Navigate to the **Prediction** tab to input data and see predictions.
    """)

elif selected == "Prediction":
    st.title('Depot Demand Prediction')
    with st.form("prediction_form"):
        st.subheader('Input Features for Demand Prediction')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            drivers = st.number_input('Number of Drivers', min_value=0)
            pickups_per_day = st.number_input('Number of Pickups per Day', min_value=0)
            inventory_level = st.number_input('Inventory Level', min_value=0)
        
        with col2:
            # depot = st.number_input('Depot', min_value=1, max_value=2)
            quantity_delivered = st.number_input('Quantity Delivered (m3)', min_value=0.0, step=0.1)
            discounts = st.number_input('Discounts', min_value=0)
        with col3:
            distance = st.number_input('Distance from Depot', min_value=0)
            stations = st.number_input('No. of Stations', min_value=0)
            
    
            
         

        submitted = st.form_submit_button('Predict Demand')

    if submitted:
        # Check if all parameters have been entered
        if not all([drivers, pickups_per_day, inventory_level, quantity_delivered, discounts, stations, distance]):
            st.warning('Please fill all the fields to make a prediction.')
        else:

            # Create a DataFrame for the model input
            data = pd.DataFrame(
                {
                'Number of Drivers': [drivers],
                'Inventory Level': [inventory_level],
                'Quantity Delivered(m3)': [quantity_delivered],
                'Distance from Depot (km)': [distance],
                'Number of Stations': [stations],
                'Discounts': [discounts],
                'Number of Pickups per Day': [pickups_per_day],    
            }
            )
            
            # st.write(predict_model(data))
            predictions = predict_model(data)
            st.write(model_category_using_y_preds(predictions))
            

            # with open('gb_m.pkl', 'rb') as f:
            #     model = pickle.load(f)
            #     prediction = model.predict(data)
            
            


 # Predictions
            

st.markdown("#### Predictions Results")
col11, col12 = st.columns(2)

# predict_results = predict_model(data)

# # Assuming the predict_model function returns predictions for different models
# for i, (model_name, result) in enumerate(predict_results.items()):
#     if i % 2 == 0:
#         col = col11  # Use column 1 for even index
#     else:
#         col = col12  # Use column 2 for odd index

#     col.warning(model_name)  # Display the model name with a warning style
#     # Assuming model_category_using_y_preds is a function to categorize predictions
#     col.success(model_category_using_y_preds(result['prediction'][0]))  # Show categorized prediction
#     col.markdown(':grey[Probability of Demand] ')  # Adding markdown for description
#     col.info(f"{result['probability'] * 100:.2f}%")  # Show probability info with formatting