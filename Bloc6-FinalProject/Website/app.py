# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from xgboost.sklearn import XGBRegressor
from pickle import load
from geopy.geocoders import Nominatim
import warnings
import plotly.express as px
import sys

# Getting the subfolder in which app.py is executed
sub_folder = sys.argv[0].replace('app.py', '')
print(sub_folder)

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(
    page_icon='🏡',
    page_title='Par[i]mmo | Estimation bien immobilier Paris intramuros',
    layout='wide')

st.sidebar.title("Par[i]mmo")

custom_styles= """
<style>
.appview-container > section:first-of-type h1 {
    font-size: calc(1.425rem + 2.1vw);
}
.main > .block-container {
    padding-top: 2rem;
}
</style>
"""
st.markdown(custom_styles, unsafe_allow_html=True)

# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: visible}
#             footer {visibility: hidden;}
#             header {visibility: visible;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

## Hack to reduce space up screen
# st.markdown("""
#         <style>
#                 .css-k1ih3n {
#                     padding-top: 1rem;
#                     padding-bottom: 10rem;
#                     padding-left: 5rem;
#                     padding-right: 5rem;
#                 }
#                 .css-1d391kg {
#                     padding-top: 3.5rem;
#                     padding-right: 1rem;
#                     padding-bottom: 3.5rem;
#                     padding-left: 1rem;
#                 }
#                 .css-1vq4p4l {
#                     padding-top: 1rem;
#                     padding-right: 1rem;
#                     padding-bottom: 1.5rem;
#                     padding-left: 1rem;
#                     }
#         </style>
#         """, unsafe_allow_html=True)

@st.cache_data
def get_full_dataset():
    return pd.read_csv(sub_folder + 'data/RealEstate_PARIS_FR_2022.csv')

model = XGBRegressor()
model.load_model(sub_folder + 'data/RealEstate_PARIS_FR_2022.xgbmodel')
preprocessor = load(open(sub_folder + 'data/preprocessor.dmp', 'rb'))
df_full_dataset = get_full_dataset()

nature_mutation = st.sidebar.selectbox('Type de vente', ('Vente', "Vente en l'état futur d'achèvement", 'Vente terrain à bâtir'))
type_local = st.sidebar.selectbox('Type de bien', ('Appartement', 'Maison'))
nb_pieces = st.sidebar.slider('Nombre de pièces', 1, 8, 2, 1)
surface = st.sidebar.slider('Surface m²', 10, 260, 80, 2)
adresse = st.sidebar.text_input('Adresse', '50 avenue des champs elysees PARIS FRANCE')


map_style = 'carto-positron'
# st.header("ESTIMATION D'UN BIEN IMMOBILIER A PARIS INTRAMUROS")
start_clicked = st.sidebar.button('👉 Calculer estimation 👈', type="primary", use_container_width=True)
if start_clicked:
    loc = Nominatim(user_agent="MyAppRE75").geocode(adresse)
    compute = (loc is not None) and ('PARIS' in adresse.upper())
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
        df_real_test['Surface_m'] = df_real_test['Surface']
        df_real_test['Surface'] = df_real_test['Surface'].apply(lambda x : '{:.0f} m²'.format(x))
        df_real_test['color'] = '#ff677f'

        # Filtering sales around the lat/lon of desired real-estate
        df_full_dataset = df_full_dataset[(abs(df_full_dataset['Latitude'] - loc.point[0]) < 0.005) &  (abs(df_full_dataset['Longitude'] - loc.point[1]) < 0.01)]
        df_full_dataset['Valeur fonciere'] = df_full_dataset['Valeur fonciere'].apply(lambda x : '{:,} €'.format(round(x)).replace(',', ' '))
        df_full_dataset['Surface_m'] = df_full_dataset['Surface']
        df_full_dataset['Surface'] = df_full_dataset['Surface'].apply(lambda x : '{:.0f} m²'.format(x))
        df_full_dataset['color'] = '#8abcde'

        fig2 = px.scatter_mapbox(df_full_dataset, lat="Latitude", lon="Longitude"
                                , zoom=15
                                , width=1200, height=700, hover_name='Valeur fonciere'
                                , hover_data={'Nature mutation':False, 'Surface_m':False, 'Surface':True, 'Nombre pieces principales':True, 'Latitude':False, 'Longitude':False}
                                , color = 'color', color_discrete_map='identity'
                                , size='Surface_m'
                                , mapbox_style=map_style
                                , opacity=1
                                )
        
        fig = px.scatter_mapbox(df_real_test, lat="Latitude", lon="Longitude"
                                , zoom=15
                                , width=1200, height=700, hover_name='Valeur fonciere'
                                , hover_data={'Nature mutation':False, 'Surface_m':False, 'Surface':True, 'Nombre pieces principales':True, 'Latitude':False, 'Longitude':False}
                                , size='Surface_m'
                                , color = 'color', color_discrete_map='identity'
                                , mapbox_style=map_style
                                , opacity=1
                                )
        
        fig.add_trace(fig2.data[0])
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        estimate = "Le bien est estimé à {:,} €".format(round(Y_real_test[0])).replace(',', ' ')
        st.header(estimate)

        st.plotly_chart(fig)

    else:
        st.header("Géolocalisation impossible, veuillez indiquer une autre adresse.")

else:
    fig = px.scatter_mapbox(pd.DataFrame({'lat':[48.86256014982167], 'lon':[2.341932519526535]})
                            , lat='lat', lon='lon'
                            , hover_data={'lat':False, 'lon':False}
                            , zoom=12
                            , width=1200, height=700
                            , mapbox_style=map_style
                            , opacity=1
                            )
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)
