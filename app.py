# importing libraries
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import shap

# reading data
df = pd.read_csv('clean_data.csv')

# selecting at top-10 frequent marks
top_10_values = df['Mark'].value_counts()[:10].index

mark = st.sidebar.selectbox(
    'Mark', top_10_values
)

model = st.sidebar.selectbox(
    'Model', df[df['Mark'] == str(mark)]['Model'].unique()
)

year = st.sidebar.number_input(
    'Year', min_value=1990, max_value=2022, value=2022
)

body_type = st.sidebar.selectbox(
    'Body type', df[df['Model'] == model]['Body_type'].unique()
)

# color
colors_dict = {
    'White': 'Ağ',
    'Black': 'Qara',
    'Blue': 'Mavi',
    'Dark blue': 'Göy',
    'Red': 'Qızılı',
    'Dark red': 'Tünd qırmızı',
    'Yellow': 'Sarı',
    'Silver': 'Gümüşü',
    'Wet Asphalt': 'Yaş Asfalt',
    'Gray': 'Boz',
    'Brown': 'Qəhvəyi',
    'Golden': 'Qırmızı',
    'Orange': 'Narıncı',
    'Purple': 'Bənövşəyi',
    'Beige': 'Bej',
    'Green': 'Yaşıl',
    'Pink': 'Çəhrayı'
}

color = st.sidebar.selectbox(
    'Color', df['Color'].unique()
)

capacity = st.sidebar.selectbox(
    'Engine capacity (L)', df[df['Model'] == model]['Capacity'].unique()
)

hp = st.sidebar.number_input(
    'Horse powers', min_value=70, max_value=700, value=300
)

engine_type = st.sidebar.selectbox(
    'Engine type', df[df['Model'] == model]['EngineType'].unique()
)

transmission = st.sidebar.selectbox(
    'Transmission', df['Transmission'].unique()
)

mileage = st.sidebar.number_input(
    'Mileage', min_value=1, max_value=500000, value=50000
)

data_dict = dict(
    Mark=mark,
    Model=model,
    Year=year,
    Body_type=body_type,
    Color=color,
    Capacity=capacity,
    HorsePowers=hp,
    EngineType=engine_type,
    Transmission=transmission,
    Mileage=mileage,
    Is_new='Xeyr'
)

data = pd.DataFrame(data_dict, index=[0])

st.title("Used car price estimator")


def make_prediction(df):

    # load OneHotEncoder and StandardScaler objects from the pickle file
    with open("objects.pkl", "rb") as f:
        objects = pickle.load(f)

    # get the OneHotEncoder and StandardScaler objects from the dictionary
    encoder = objects["encoder"]
    scaler = objects["scaler"]

    # set numeric columns
    numeric_columns = ['Mileage', 'HorsePowers']

    # scale the numerical columns
    scaled_data = scaler.transform(df[numeric_columns])
    df[numeric_columns] = scaled_data

    # set the categorical columns
    categorical_columns = ['Mark', 'Model', 'Body_type',
                           'Color', 'EngineType', 'Transmission']

    encoded_data = encoder.transform(df[categorical_columns])
    df_encoded = pd.DataFrame(
        encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

    # concatenate encoded data with actual
    df = pd.concat([df.drop(categorical_columns, axis=1), df_encoded], axis=1)

    # encode Is_new column to 0 and 1 values
    df['Is_new'] = df['Is_new'].map({'Bəli': 1, 'Xeyr': 0})

    # load the model from the file
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    df.drop('Is_new', axis=1, inplace=True)

    # make prediction
    pred = model.predict(df)

    # check feature importance
    # explainer = shap.TreeExplainer(model)

    # shap_values = explainer.shap_values(df)
    # shap.initjs()
    # fig2 = shap.force_plot(explainer.expected_value[0], shap_values[0],
    #                        df, feature_names=df.columns)
    # shap_html = f"<head>{shap.getjs()}</head><body>{fig2.html()}</body>"
    # st.write('Explanation')
    # st.components.v1.html(shap_html)

    return np.round(pred[0])


st.subheader(f"Your car estimated price is {make_prediction(data)} $")
