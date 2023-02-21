# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from multi_threaded_supervised_ml import MultiThreadedClassifier, MultiThreadedRegressor, CLASSIFIERS, REGRESSORS, get_numeric_categorical_features

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, fetch_california_housing 

import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The ML Algorithm Comparison App', layout='wide')
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
               .css-18e3th9 {
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


# Model building
def build_model():

    X = df.drop(columns=target_column, axis=1)
    Y = df.loc[:,target_column]

    # # Splitting testing and training sets
    if predict_class == 'Classifier':
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed_number, stratify=Y)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed_number)

    # Data normalization
    preprocessor = None
    if normalization_type == 'Custom':
        numeric_features, categorical_features = get_numeric_categorical_features(X_train)
        
        # numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                                  ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(sparse_threshold=0, 
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ])
    elif normalization_type == 'None':
        preprocessor = 'forced_none'

    # Building models
    if predict_class == 'Regressor':
        class_= MultiThreadedRegressor
        models_ = REGRESSORS
        y_label_encoder = False
    else:
        class_= MultiThreadedClassifier
        models_ = CLASSIFIERS
        y_label_encoder = True
    models = class_(seed_number, models_, multithread_num_workers, stacking_estimators_choice, [first_stacking_estimator]+[second_stacking_estimator]+[third_stacking_estimator]+[final_stacking_estimator]
                                        , voting_estimators_choice, [first_voting_estimator]+[second_voting_estimator]+[third_voting_estimator]+[fourth_voting_estimator])
    
    print('\nTraining and testing sets fit:')
    models.multithread_fit(X_train, X_test, Y_train, Y_test, preprocessor, metric_name, y_label_encoder)

    with tab_ds:
        with tab_ds_trans:
            if models.preprocessor is not None:
                with st.expander('Preprocessor used'):
                    st.write(models.preprocessor.transformers)
            st.write('Trained dataset')
            st.write(models.X_train.head(10))

    if models.scores_test.shape[0] > 0 and models.scores_test.shape[0] > 0:
        with tab_ml:
            with tab_ml_results:
                st.subheader('Table of Models Performance')
            
                col1, col2 = st.columns(2)
            
                with col1:
                    st.write('Test set')
                    st.write(models.scores_test)
                    st.markdown(filedownload(models.scores_test,'test.csv'), unsafe_allow_html=True)
            
                with col2:
                    st.write('Training set')
                    st.write(models.scores_train)
                    st.markdown(filedownload(models.scores_train,'training.csv'), unsafe_allow_html=True)
        
                # Displaying results
                st.subheader(f'Plot of Models Performance ({metric_name})')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write('Testing results')
                    models.scores_test[metric_name] = [0 if i < 0 else i for i in models.scores_test[metric_name] ]
                    plt.figure(figsize=(10, 10))
                    ax1 = sns.barplot(y=models.scores_test.index, x=models.scores_test[metric_name], data=models.scores_test, palette="rocket")
                    ax1.set(xlim=(0, 1))
                    st.pyplot(plt)
                    st.markdown(pdfdownload(plt,'plot-testing.pdf'), unsafe_allow_html=True)
                with col2:
                    st.write('Training results')
                    models.scores_train[metric_name] = [0 if i < 0 else i for i in models.scores_train[metric_name] ]
                    plt.figure(figsize=(10, 10))
                    ax1 = sns.barplot(y=models.scores_train.index, x=models.scores_train[metric_name], data=models.scores_train, palette="rocket")
                    ax1.set(xlim=(0, 1))
                    st.pyplot(plt)
                    st.markdown(pdfdownload(plt,'plot-training.pdf'), unsafe_allow_html=True)
                with col3:
                    st.write('Calculation time for training')
                    models.scores_train["Time Taken"] = [0 if i < 0 else i for i in models.scores_train["Time Taken"] ]
                    plt.figure(figsize=(10, 10))
                    sns.barplot(y=models.scores_train.index, x=models.scores_train["Time Taken"], data=models.scores_train, palette="viridis")
                    st.pyplot(plt)
                    st.markdown(pdfdownload(plt,'plot-calculation-time.pdf'), unsafe_allow_html=True)
        
        with tab_ml:
            with tab_ml_params:
                # Displaying ML parameters
                best_models_count = min(5, models.scores_test.shape[0])
                st.subheader(f'Parameters of {best_models_count} best testing models')
                for i in range(best_models_count):
                    model_name = models.scores_test.index[i]
                    st.subheader(f'👉{model_name}👈')
                    st.text("Model Params")
                    best_model = models.params[model_name]['model']
                    st.write(best_model.get_params())
    else:
        st.subheader(":red[!!! Issues occured during Machine Learning fits !!!]")

    with tab_ml:
        with tab_ml_errors:
            st.subheader('Models execution errors')
            st.write(models.df_errors)

def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def pdfdownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/pdf;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

@st.cache_data()
def get_df_from_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

#-----------------------------------------------------------#
# Sidebar - Collects user input features into dataframe
#-----------------------------------------------------------#

df = None
with st.sidebar:
    st.subheader('1. Upload your cleaned-up CSV file')

    col1, col2 = st.columns(2)
    with col1:
        use_sample_dataset = st.checkbox("Use inner dataset?", value=False)
    with col2:
        sample_dataset = st.radio("Dataset", ["Breast Cancer", "California Housing"], horizontal=True, disabled=not use_sample_dataset, label_visibility="collapsed")
    if not use_sample_dataset:
        uploaded_file = st.sidebar.file_uploader("Upload your cleaned-up CSV file", type=["csv"])
        if uploaded_file is not None:
            df = get_df_from_csv(uploaded_file)
    else:
        if use_sample_dataset:
            if sample_dataset == "Breast Cancer":
                dataset = load_breast_cancer()
            else:
                dataset = fetch_california_housing ()
            uploaded_file = None
            X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            Y = pd.Series(dataset.target, name='response')
            df = pd.concat([X,Y], axis=1)

    st.subheader('2. Set the machine learning parameters')

    col1, col2 = st.columns(2)
    with col1:
        if df is not None:
            target_column = st.selectbox('Target column', df.columns.to_list(), len(df.columns)-1)
        else:
            st.write('Target column...')

    col1, col2 = st.columns(2)
    with col1:
        predict_class = st.selectbox('Predict class', ('Classifier', 'Regressor'), (1 if use_sample_dataset and sample_dataset=="California Housing" else 0))
            
        normalization_type = st.selectbox("Normalization", ('None', 'Inner', 'Custom'))
        
    with col2:    
        if predict_class == 'Classifier':
            metric_name = st.selectbox('Metric', ('PR AUC', 'F1 Score', 'Balanced Accuracy', 'Accuracy', 'ROC AUC'))
        else:
            metric_name = st.selectbox('Metric', ('Adjusted R-Squared', 'R-Squared', 'RMSE'))
        sample_sizes = ['All','1 000 000', '100 000', '10 000', '1 000', '100']
        sample_size = st.selectbox('Sample size', sample_sizes).replace(' ', '')
        
    first_stacking_estimator = "None"
    second_stacking_estimator = "None"
    third_stacking_estimator = "None"
    final_stacking_estimator = "None"
    
    first_voting_estimator = "None"
    second_voting_estimator = "None"
    third_voting_estimator = "None"
    fourth_voting_estimator = "None"

    stacking_estimators_choice = st.selectbox(f"Stacking{predict_class}", ('No', 'Yes', 'Only'))

    if predict_class == "Classifier":
        possible_estimators = CLASSIFIERS
    else:
        possible_estimators = REGRESSORS
        
    # col1, col2 = st.columns(2)
    if stacking_estimators_choice != "No":
        col1, col2 = st.columns(2)
        with col1:
            first_stacking_estimator = st.selectbox('1st Stacking estimator', [i[0] for i in possible_estimators])
            third_stacking_estimator = st.selectbox('3rd Stacking estimator', ['None'] + [i[0] for i in possible_estimators])
        with col2:
            second_stacking_estimator = st.selectbox('2nd Stacking estimator', ['None'] + [i[0] for i in possible_estimators])
            final_stacking_estimator = st.selectbox('final Stacking estimator', [i[0] for i in possible_estimators], index=18) 
        
    voting_estimators_choice = st.selectbox(f"Voting{predict_class}", ('No', 'Yes', 'Only'))
    # col1, col2 = st.columns(2)
    if voting_estimators_choice != "No":
        col1, col2 = st.columns(2)
        with col1:
            first_voting_estimator = st.selectbox('1st Voting estimator', [i[0] for i in possible_estimators])
            third_voting_estimator = st.selectbox('3rd Voting estimator', ['None'] + [i[0] for i in possible_estimators])
        with col2:
            second_voting_estimator = st.selectbox('2nd Voting estimator', ['None'] + [i[0] for i in possible_estimators])
            fourth_voting_estimator = st.selectbox('4th Voting estimator', ['None'] + [i[0] for i in possible_estimators]) 
        
    
    multithread_num_workers = st.sidebar.slider('Set the multithread workers number', 1, 8, 8, 1)
    
    split_size_selector = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    split_size = round(1 - split_size_selector/100, 2)
    
    seed_number = st.sidebar.slider('Set the random seed number', 0, 100, 0, 1)

#---------------------------------#
# Main panel

if df is not None:
    start_clicked = st.button('👉 Press to start calculations 👈')
    
    tab_ds, tab_ml = st.tabs(["DATASET", "MACHINE LEARNING"])    
    with tab_ds:
        tab_ds_details, tab_ds_trans = st.tabs(["Details", "Transformed Dataset"])

    with tab_ml:
        tab_ml_results, tab_ml_params, tab_ml_errors = st.tabs(["Results", "Models Params", "ML errors"])
    
    if sample_size != 'All':
        df = df.head(int(sample_size))
    
    with tab_ds_details:
        st.markdown('**Dataset**')
        st.write(df.head(10))
    
        st.markdown('**Dataset dimension**')
        col1, col2 = st.columns(2)
        with col1:
            st.info(f'ROW COUNT: {df.shape[0]}')
        with col2:
            st.info(f'COLUMN COUNT: {df.shape[1]}')
    
        st.markdown('**Dataset informations**')
        st.write(df.describe(include='all'))
        
        st.markdown('**Values details**')
        st.write(pd.concat([df.isnull().any().to_frame('Nulls?'), df.nunique().to_frame('Unique count')], axis=1))    

    if start_clicked:
        build_model()
else:
    st.write('Waiting for a dataset to be loaded ...')

