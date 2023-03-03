# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import matplotlib.pyplot as plt
import sys
from PIL import Image

# Getting the subfolder in which app.py is executed
sub_folder = sys.argv[0].replace('app.py', '')

#---------------------------------#
# Page layout
## Page expands to full width
im = Image.open(sub_folder + "getaround.ico")
st.set_page_config(
    page_icon=im,
    page_title='Getaround delay dashboard',
    layout='wide')

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

col1, col2 = st.columns(2)
with col1:
    st.image(sub_folder + 'getaround.png')
with col2:    
    st.title("DELAY DASHBOARD")

@st.cache_data
def get_full_dataset():
    return pd.read_excel('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx')

df_delay = get_full_dataset()

st.markdown('**Dataset**')
st.write(df_delay.head(5))

st.header("Comparison between Mobile and Connect rentals")
fig = px.histogram(df_delay.sort_values('checkin_type'), y='checkin_type', color='checkin_type', color_discrete_sequence=px.colors.qualitative.Prism)
st.plotly_chart(fig, use_container_width=True)

st.header("Comparison on state between mobile and connect rental")
df_delay_connect = df_delay[df_delay.checkin_type=='connect']
df_delay_mobile = df_delay[df_delay.checkin_type=='mobile']
df_delay_connect_cancel = df_delay_connect[df_delay_connect.state=='canceled']
df_delay_mobile_cancel = df_delay_mobile[df_delay_mobile.state=='canceled']
col1, col2 = st.columns(2)
with col1:
    st.subheader("Percentage of cancelations for connect rentals is " + str(round(df_delay_connect_cancel.shape[0]/df_delay_connect.shape[0]*100)) + "%")
with col2:
    st.subheader("Percentage of cancelations for mobile rentals is " + str(round(df_delay_mobile_cancel.shape[0]/df_delay_mobile.shape[0]*100)) + "%")
fig = px.bar(df_delay, orientation='h', y='checkin_type', color='state', color_discrete_map={'ended': 'blue','canceled': 'red'})
st.plotly_chart(fig, use_container_width=True)

st.header("Distribution of delays following checkin type")
df_delay_without_outliers = df_delay[abs(df_delay.delay_at_checkout_in_minutes)<=500]
fig = px.histogram(df_delay_without_outliers, x='delay_at_checkout_in_minutes', color='checkin_type', barmode='overlay', color_discrete_sequence=px.colors.qualitative.Prism)
st.plotly_chart(fig)

st.header("Threshold of checkout delay")
df_positive_delay_without_outliers = df_delay_without_outliers[df_delay_without_outliers.delay_at_checkout_in_minutes > 0]
df_mobile_with_thresold = df_positive_delay_without_outliers[df_positive_delay_without_outliers.checkin_type=='mobile']
df_connect_with_thresold = df_positive_delay_without_outliers[df_positive_delay_without_outliers.checkin_type=='connect']

impacted_mobile_count = []
impacted_mobile_percentage = []
impacted_connect_count = []
impacted_connect_percentage = []
arange = range(10, 500, 10)
for i in arange:
    count = df_mobile_with_thresold[df_mobile_with_thresold.delay_at_checkout_in_minutes > i].shape[0]
    impacted_mobile_count.append(count)
    impacted_mobile_percentage.append((count / df_mobile_with_thresold.shape[0])*100)
    count = df_connect_with_thresold[df_connect_with_thresold.delay_at_checkout_in_minutes > i].shape[0]
    impacted_connect_count.append(count)
    impacted_connect_percentage.append((count / df_connect_with_thresold.shape[0])*100)
    
col1, col2 = st.columns(2)
with col1:
    fig, ax1 = plt.subplots(1, 1)    
    st.subheader("Count of impacted rentals following a threshold on allowed checkout delay")
    ax1.plot(arange, impacted_connect_count)
    ax1.plot(arange, impacted_mobile_count)
    ax1.legend(['connect', 'mobile'], labelcolor="white", fontsize="large")
    ax1.set_xlabel('Threshold of checkout delay', fontsize="large")
    ax1.xaxis.label.set_color('white')
    st.plotly_chart(fig, use_container_width=True)
    
with col2:
    fig, ax1 = plt.subplots(1, 1)     
    st.subheader("Percentage of impacted rentals following a threshold on allowed checkout delay")
    ax1.plot(arange, impacted_connect_percentage)
    ax1.plot(arange, impacted_mobile_percentage)
    ax1.legend(['connect', 'mobile'], labelcolor="white", fontsize="large")
    ax1.set_xlabel('Threshold of checkout delay', fontsize="large")
    ax1.xaxis.label.set_color('white')
    st.plotly_chart(fig, use_container_width=True)
    