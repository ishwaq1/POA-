import streamlit as st
import pandas as pd
from joblib import load 
import os
import numpy as np
from xgboost import XGBClassifier  
import pickle

# from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split

from lime import lime_tabular

@st.cache_data
def  read_data():
    return pd.read_csv('https://raw.githubusercontent.com/ishwaq1/MLdata-/main/depot%20data.csv')


# @st.experimental_memo(allow_output_mutation=True)
def model_load():
    filepath = r"model_gb.pkl"  # Use a raw string for the file path
    loaded_models = load(filepath)  # Load the model from the specified path
    return loaded_models



def predict_model(data):
    with open('gb_m.pkl', 'rb') as f:
        model = pickle.load(f)
    # data = data.reset_index(drop=True)
    predictions = model.predict(data)
    
    return predictions

    # Categorization function
def model_category_using_y_preds(prediction):
    if prediction > 60:
        return 'High Demand'
    else:
        return 'Low Demand'
    

# Create a button to download the model
def download_objects(model_path):
    
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    st.sidebar.download_button(
        label="Click to download",
        data=model_bytes,
        file_name=os.path.basename(model_path),
        mime="application/octet-stream"
    )

def load_and_display_data():
    # Load the dataset from the specified URL
    df = pd.read_csv("https://raw.githubusercontent.com/ishwaq1/MLdata-/main/depot%20data.csv")
    
    # Set a threshold for what constitutes high demand
    some_threshold = 100  # Adjust this threshold based on your analysis needs

    # Create a new column in the DataFrame based on the 'Demand' column
    df['HighDemand'] = np.where(df['Demand'] > some_threshold, 'High', 'Low')

    # Display the DataFrame in Streamlit
    st.write("Data Overview:", df.head())  # Show the first few rows of the DataFrame
    st.write("Demand Distribution:", df['HighDemand'].value_counts())  # Display the distribution of demand categories

    # Optionally, use a bar chart to visualize the distribution of 'HighDemand'
    st.bar_chart(df['HighDemand'].value_counts())

# This block to ensure the function runs in the Streamlit environment when executed as a script
if __name__ == "__main__":
    load_and_display_data()

    
from sklearn.model_selection import train_test_split

def train_test(df):
    """Splits the data into training and testing sets."""
    x = df.drop(columns=['HighDemand'])  # Adjust 'HighDemand' to your target column name
    Y = df['HighDemand']  # Adjust 'HighDemand' to your target column name
    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.3)
    return x_train, x_test, y_train, y_test
   


def predict_fn(x):
    loaded_models = model_load()
    for model, result in zip(loaded_models):
        if result['model_name']== 'XGBoosting':
            model = model.predict_proba(x)
    return model


def download_model(model_path):
    """Creates a button in the Streamlit sidebar for downloading a model file."""
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    st.sidebar.download_button(
        label="Click to download model",
        data=model_bytes,
        file_name=os.path.basename(model_path),
        mime="application/octet-stream"
    )


    def load_and_display_data():
    # Load the dataset from the specified URL
        df = pd.read_csv("https://raw.githubusercontent.com/ishwaq1/MLdata-/main/depot%20data.csv")
        
        # Set a threshold for what constitutes high demand
        some_threshold = 100  # Adjust this threshold based on your analysis needs

        # Create a new column in the DataFrame based on the 'Demand' column
        df['HighDemand'] = np.where(df['Demand'] > some_threshold, 'High', 'Low')

        # Display the DataFrame in Streamlit
        st.write("Data Overview:", df.head())  # Show the first few rows of the DataFrame
        st.write("Demand Distribution:", df['HighDemand'].value_counts())  # Display the distribution of demand categories

        # Optionally, use a bar chart to visualize the distribution of 'HighDemand'
        st.bar_chart(df['HighDemand'].value_counts())

    # Assuming this function is called in a Streamlit environment
    if __name__ == "__main__":
        load_and_display_data()




# @st.experimental_memo(allow_output_mutation=True)
def lime_explainer(df, instance_index):
    """Generates LIME explanations for a model's predictions."""
    x_train, x_test, y_train, y_test = train_test(df)
    
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(x_train),
        feature_names=x_train.columns.tolist(),
        class_names=['Low Demand', 'High Demand'],
        mode='classification',
        random_state=42
    )
    
    instance = x_test.iloc[[int(instance_index)]]
    explanation = explainer.explain_instance(instance.values[0], predict_fn, num_features=len(x_train.columns))
    html= explanation.as_html()
    
    return html