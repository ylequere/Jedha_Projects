# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from xgboost.sklearn import XGBRegressor
from pickle import load
from geopy.geocoders import Nominatim
import warnings
import plotly.express as px


#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Estimation bien immobilier Paris intramuros', layout='wide')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: visible}
            footer {visibility: hidden;}
            header {visibility: visible;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

## Hack to reduce space up screen
st.markdown("""
        <style>
                .css-k1ih3n {
                    padding-top: 1rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
                .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
                .css-1vq4p4l {
                    padding-top: 1rem;
                    padding-right: 1rem;
                    padding-bottom: 1.5rem;
                    padding-left: 1rem;
                    }
        </style>
        """, unsafe_allow_html=True)

def get_model():
    return XGBRegressor().load_model('./RealEstate_PARIS_FR_2022.xgbmodel')

def get_preprocessor():
    return load(open('./preprocessor.dmp', 'rb'))

def get_geoloc():
    return Nominatim(user_agent="MyAppRE75")

@st.cache_data
def get_full_dataset():
    return pd.read_csv('./TempFiles/RealEstate_PARIS_FR_2022.csv')

model = XGBRegressor()
model.load_model('./RealEstate_PARIS_FR_2022.xgbmodel')
preprocessor = get_preprocessor()
df_full_dataset = get_full_dataset()

nature_mutation = st.sidebar.selectbox('Nature mutation', ('Vente', "Vente en l'état futur d'achèvement", 'Vente terrain à bâtir'))
type_local = st.sidebar.selectbox('Type local', ('Appartement', 'Maison'))
nb_pieces = st.sidebar.slider('Nombre pieces principales', 1, 8, 2, 1)
surface = st.sidebar.slider('Surface', 10, 260, 80, 2)
adresse = st.sidebar.text_input('Adresse', '8 avenue des champs elysees PARIS FRANCE')

# st.header("ESTIMATION D'UN BIEN IMMOBILIER A PARIS INTRAMUROS")
start_clicked = st.sidebar.button('👉 Calculer estimation 👈')
if start_clicked:
    loc = get_geoloc().geocode(adresse)
    compute = loc is not None
    if compute:
        df_real_test = pd.DataFrame({'Nature mutation':[nature_mutation]
                                     , 'Type local':[type_local]
                                     , 'Nombre pieces principales': [nb_pieces]
                                     , 'Surface': [surface]
                                     , 'Latitude': [loc.point[0]]
                                     , 'Longitude': [loc.point[1]]
                                     })

        X_real_test = preprocessor.transform(df_real_test)
        Y_real_test = model.predict(X_real_test)    

        df_real_test['Valeur fonciere'] = Y_real_test
        df_real_test['Valeur fonciere'] = df_real_test['Valeur fonciere'].apply(lambda x : '{:,} €'.format(round(x)).replace(',', ' '))
        df_real_test['color'] = 'green'

        # Filtering sales around the lat/lon of desired real-estate
        df_full_dataset = df_full_dataset[(abs(df_full_dataset['Latitude'] - loc.point[0]) < 0.005) &  (abs(df_full_dataset['Longitude'] - loc.point[1]) < 0.01)]
        df_full_dataset['Valeur fonciere'] = df_full_dataset['Valeur fonciere'].apply(lambda x : '{:,} €'.format(round(x)).replace(',', ' '))
        df_full_dataset['color'] = 'blue'

        fig2 = px.scatter_mapbox(df_full_dataset, lat="Latitude", lon="Longitude"
                                , zoom=15
                                , width=1200, height=700, hover_name='Valeur fonciere'
                                , hover_data=['Surface', 'Nombre pieces principales']
                                , color = 'color', color_discrete_map='identity'
                                , size='Surface'
                                , mapbox_style="open-street-map"
                                )
        
        fig = px.scatter_mapbox(df_real_test, lat="Latitude", lon="Longitude"
                                , zoom=15
                                , width=1200, height=700, hover_name='Valeur fonciere'
                                , hover_data=['Surface', 'Nombre pieces principales']
                                , size='Surface'
                                , color = 'color', color_discrete_map='identity'
                                , mapbox_style="open-street-map", opacity=1
                                )
        
        fig.add_trace(fig2.data[0])
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        estimate = "L'estimation du bien est {:,} €".format(round(Y_real_test[0])).replace(',', ' ')
        st.header(estimate)

        st.plotly_chart(fig)

    else:
        st.header("Géolocalisation impossible, veuiller indiquer une autre adresse.")
