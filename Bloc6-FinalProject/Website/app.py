# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from xgboost.sklearn import XGBRegressor
from pickle import load
from geopy.geocoders import Nominatim
import plotly.express as px
import sys
import altair as alt

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
    padding-left: 1rem;
    padding-right: 1rem;
}

.main > footer {
    display: none;
}
</style>
"""
st.markdown(custom_styles, unsafe_allow_html=True)

@st.cache_data
def get_full_dataset():
    return pd.read_csv(sub_folder + 'data/TempFiles/RealEstate_PARIS_FR_2022.csv')

model = XGBRegressor()
model.load_model(sub_folder + 'data/RealEstate_PARIS_FR_2022.xgbmodel')
preprocessor = load(open(sub_folder + 'data/preprocessor.dmp', 'rb'))
df_full_dataset = get_full_dataset()

nature_mutation = st.sidebar.selectbox('Type de vente', ('Vente', "Vente en l'état futur d'achèvement"))
type_local = 'Appartement'
nb_pieces = st.sidebar.slider('Nombre de pièces', 1, 8, 2, 1)
surface = st.sidebar.slider('Surface m²', 10, 260, 80, 2)
adresse = st.sidebar.text_input('Adresse', '50 avenue des champs elysees PARIS FRANCE')

map_style = 'carto-positron'
# st.header("ESTIMATION D'UN BIEN IMMOBILIER A PARIS INTRAMUROS")
start_clicked = st.sidebar.button('👉 Calculer estimation 👈', type="primary", use_container_width=True)
if start_clicked:
    loc = Nominatim(user_agent="MyAppRE75").geocode(adresse, language="fr")
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
        df_full_dataset['Valeur fonciere raw'] = df_full_dataset['Valeur fonciere']
        df_full_dataset['Valeur fonciere'] = df_full_dataset['Valeur fonciere'].apply(lambda x : '{:,} €'.format(round(x)).replace(',', ' '))
        df_full_dataset['Surface_m'] = df_full_dataset['Surface']
        df_full_dataset['Surface'] = df_full_dataset['Surface'].apply(lambda x : '{:.0f} m²'.format(x))
        df_full_dataset['color'] = '#8abcde'

        fig2 = px.scatter_mapbox(df_full_dataset, lat="Latitude", lon="Longitude"
                                , zoom=15
                                , height=600, hover_name='Valeur fonciere'
                                , hover_data={'Nature mutation':False, 'Surface_m':False, 'Surface':True, 'Nombre pieces principales':True, 'Latitude':False, 'Longitude':False}
                                , color = 'color', color_discrete_map='identity'
                                , size='Surface_m'
                                , mapbox_style=map_style
                                , opacity=1
                                )
        
        fig = px.scatter_mapbox(df_real_test, lat="Latitude", lon="Longitude"
                                , zoom=15
                                , height=600, hover_name='Valeur fonciere'
                                , hover_data={'Nature mutation':False, 'Surface_m':False, 'Surface':True, 'Nombre pieces principales':True, 'Latitude':False, 'Longitude':False}
                                , size='Surface_m'
                                , color = 'color', color_discrete_map='identity'
                                , mapbox_style=map_style
                                , opacity=1
                                )
        
        fig.add_trace(fig2.data[0])
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        estimate_raw = round(Y_real_test[0])
        estimate = "{:,} €".format(estimate_raw).replace(',', ' ')
        st.header("Le bien est estimé à " + estimate)

        col1, col2 = st.columns([4, 1])

        with col1:
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader('Secteur')

            secteur_valeur_fonciere_mean = df_full_dataset['Valeur fonciere raw'].mean()
            valeur_fonciere_delta = estimate_raw - secteur_valeur_fonciere_mean

            secteur_surface_m_mean = df_full_dataset['Surface_m'].mean()
            surface_m_delta = surface - secteur_surface_m_mean

            secteur_valeur_surface_m_mean = secteur_valeur_fonciere_mean / secteur_surface_m_mean
            valeur_surface_m_delta = (estimate_raw / surface) - secteur_valeur_surface_m_mean

            st.metric(label="Prix moyen au m²"
                    , value="{:,} €".format(round(secteur_valeur_surface_m_mean)).replace(',', ' ')
                    , delta="{:,} €".format(round(valeur_surface_m_delta)).replace(',', ' ')
                    )

            st.metric(label="Surface moyenne"
                      , value='{:.00f} m²'.format(round(secteur_surface_m_mean, 2))
                      , delta='{:.00f} m²'.format(round(surface_m_delta, 2))
                      )

            # https://altair-viz.github.io/user_guide/customization.html
            histogram = alt.Chart(
                df_full_dataset).mark_bar(size=20).encode(
                    alt.X('Nombre pieces principales:N', title='Pièces principales', axis=alt.Axis(format='.1', tickMinStep=1, labelAngle=0), scale=alt.Scale(zero=False)),
                    alt.Y('count()', title='Nombre de ventes'),
                    color=alt.Color('Nombre pieces principales', legend=None),
                    tooltip={"field": "__count", "title": "Nombre de ventes"}
                ).configure_axis(
                   grid=False
                ).configure_view(
                   strokeWidth=0
                ).interactive()
            st.altair_chart(histogram, use_container_width=True)
    else:
        st.header("Géolocalisation impossible, veuillez indiquer une autre adresse.")
else:
    fig = px.scatter_mapbox(pd.DataFrame({'lat':[48.86256014982167], 'lon':[2.341932519526535]})
                            , lat='lat', lon='lon'
                            , hover_data={'lat':False, 'lon':False}
                            , zoom=12
                            , height=600
                            , mapbox_style=map_style
                            , opacity=1
                            )
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)
