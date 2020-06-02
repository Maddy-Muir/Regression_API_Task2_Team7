"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    
    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
   
    feature_vector_df = feature_vector_df.drop(['Vehicle Type', 'Temperature', 
                                                'Precipitation in millimeters' ], 
                                                 axis = 1)
    feature_vector_df = feature_vector_df.drop(['User Id', 'Platform Type'], 
                                                 axis = 1)

    def dummy_encode_columns(input_df, column_name):
        dummy_df = pd.get_dummies(input_df, columns = [column_name], drop_first = True)
        return dummy_df

    feature_vector_df = dummy_encode_columns(feature_vector_df, 'Personal or Business')

    feature_vector_df = feature_vector_df.drop(['Placement - Weekday (Mo = 1)',
                                  'Confirmation - Weekday (Mo = 1)',
                                  'Arrival at Pickup - Weekday (Mo = 1)',
                                  'Pickup - Weekday (Mo = 1)'], axis = 1)

    feature_vector_df = feature_vector_df.drop(['Confirmation - Day of Month',
                                  'Placement - Day of Month',
                                  "Arrival at Pickup - Day of Month",
                                  'Pickup - Day of Month'], axis = 1)
    
    testing_time_cols = ['Placement - Time', 'Confirmation - Time', 'Arrival at Pickup - Time', 
                      'Pickup - Time']

    for time in testing_time_cols:
        feature_vector_df[time] = pd.to_datetime(feature_vector_df[time])

    feature_vector_df['Time Difference - Placement to Confirmation'] = (
        (feature_vector_df['Confirmation - Time'] - feature_vector_df['Placement - Time'])
        .dt.total_seconds()
    )

    feature_vector_df['Time Difference - Confirmation to Arrival at Pickup'] = (
        (feature_vector_df['Arrival at Pickup - Time'] - feature_vector_df['Confirmation - Time'])
        .dt.total_seconds()
    )

    feature_vector_df['Time Difference - Arrival at Pickup to Pickup'] = (
        (feature_vector_df['Pickup - Time'] - feature_vector_df['Arrival at Pickup - Time'])
        .dt.total_seconds()
    ) 

    feature_vector_df = feature_vector_df.drop(['Placement - Time', 'Confirmation - Time', 
                                'Arrival at Pickup - Time', 'Pickup - Time'], axis = 1)


    def extract_id(input_df):
        input_df['Rider Id'] = input_df['Rider Id'].str.extract(r"([0-9]+)").astype(int)
        return input_df

    extract_id(feature_vector_df)

    final_features = ['Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long',
       'Rider Id', 'Age', 'Average_Rating',
       'Time Difference - Placement to Confirmation',
       'Time Difference - Arrival at Pickup to Pickup', 
       'Distance (KM)']

    feature_vector_df = feature_vector_df[final_features]

    return feature_vector_df

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    
    # Format as list for output standerdisation.
    return prediction.tolist()
