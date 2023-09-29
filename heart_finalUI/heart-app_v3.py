import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Create a page dropdown
page = st.sidebar.selectbox("""
Hello there! I'll guide you!
Please select Menu""", ["Main Page", "Compare Models"])

#-------------------------------------------------


if page == "Main Page":

    ### INFO
    #st.title("Hello, welcome to sales predictor!")
    # st.write("""
    # This application predicts sales for the next 10 days with 3 different models
    
    # # Sales drivers used in prediction: 
    
    # - Date: date format time feature
    # - col1: categorical feature 
    # - col2: second categorical feature
    # - col3: third categorical feature
    # - target: target variable to be predicted
      
   # """)


    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
        img_to_bytes(r"C:\Users\sndip\Downloads\streamlit_heart_final\streamlit_heart\heart_finalUI\heart-pulse-life-wallpaper-preview.jpg")
    )
    st.markdown(
        header_html, unsafe_allow_html=True,
    )
    
    
    
    st.write("""
    #                          Cardiac Desease Prediction
    """)
    # Customer churn is defined as the loss of customers after a certain period of time. Companies are interested in targeting customers
    # who are likely to churn. They can target these customers with special deals and promotions to influence them to stay with
    # the company. 
    
    # This app predicts the probability of a customer churning using Telco Customer data. Here
    # customer churn means the customer does not make another purchase after a period of time. 
    
    
    
    
    
    df_selected = pd.read_csv(r'C:\Users\sndip\Downloads\streamlit_heart_final\streamlit_heart\heart_finalUI\final_heart_data_v1.csv')
    df_selected_all = df_selected.copy()
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'
        return href
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown(filedownload(df_selected_all), unsafe_allow_html=True)
    
    
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            age = st.sidebar.slider('age', 1.0,80.0, 1.0
                                    )
            sex = st.sidebar.selectbox('sex',(1,0))
            cp = st.sidebar.selectbox('cp',(1.0, 2.0, 3.0, 4.0))
            trestbps = st.sidebar.slider('trestbps', 0.0,200.0, 20.0)
            chol = st.sidebar.slider('chol', 0.0,605.0, 100.0)
            fbs = st.sidebar.selectbox('fbs',(0.0,1.0))
            restecg = st.sidebar.selectbox('restecg',(0.0,1.0,2.0))
            thalach = st.sidebar.slider('thalach', 60.0,205.0, 70.0)
            exang = st.sidebar.selectbox('exang',(0.0,1.0))
            oldpeak = st.sidebar.slider('oldpeak', -2.6,6.5, 1.0)
            slope = st.sidebar.selectbox('slope',(1.0,2.0,3.0))
            print(type(sex))
            print(type(trestbps))
    
    
            data = {'age':[age], 
                    'sex':[sex], 
                    'cp':[cp], 
                    'trestbps':[trestbps],
                    'chol':[chol],
                    'fbs':[fbs],
                    'restecg':[restecg],
                    'thalach':[thalach],
                    'exang':[exang],
                    'oldpeak':[oldpeak],
                    'slope':[slope]
                    }
            
            features = pd.DataFrame(data)
            return features
        input_df = user_input_features()
        
        
    
    
    
    churn_raw = pd.read_csv(r'C:\Users\sndip\Downloads\streamlit_heart_final\streamlit_heart\heart_finalUI\final_heart_data_v1.csv')
    
    #churn_raw.info()
    #churn_raw.isnull().any()
    #churn_raw.shape
    #input_df.shape
    
    churn_raw.fillna(0, inplace=True)
    churn = churn_raw.drop(columns=['condition'])
    df = pd.concat([input_df,churn],axis=0)
    df = df[:1] # Selects only the first row (the user input data)
    df.fillna(0, inplace=True)
    
    #df.shape
    #df.isna().any()
    # churn.info()
    # df = churn.copy()
    
    
    
    # encode = ['sex','cp','fbs','restecg','exang','slope']
    
    # from sklearn.preprocessing import LabelEncoder
    # lb=LabelEncoder()
    # for i in encode:
    #     df[i]=lb.fit_transform(df[i])
    
    # for col in encode:
    #     dummy = pd.get_dummies(df[col], prefix=col)
    #     df = pd.concat([df,dummy], axis=1)
    #     del df[col]
    #df = df[:1] # Selects only the first row (the user input data)
    #df.fillna(0, inplace=True)
    
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
           'exang', 'oldpeak', 'slope']
    df = df[features]
    #df.shape
    
    
    
    # Displays the user input features
    st.subheader('User Input features')
    print(df.columns)
    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        st.write(df)
    
    # Reads in saved classification model
    load_clf = pickle.load(open('./heart_clf.pkl', 'rb'))
    
    # Apply model to make predictions
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)
    import plotly.graph_objects as go
    
    def plot_pie_chart(probabilities):
            fig = go.Figure(
                data=[go.Pie(
                        labels=list(set(churn_raw.condition)),
                        values=probabilities[0]
                )]
            )
            fig = fig.update_traces(
                hoverinfo='label+percent',
                textinfo='value',
                textfont_size=15
            )
            return fig
        
    st.markdown(
                """
                <style>
                .header-style {
                    font-size:25px;
                    font-family:sans-serif;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown(
                """
                <style>
                .font-style {
                    font-size:20px;
                    font-family:sans-serif;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown(
                '<p class="header-style"> Heart Desease Predictions </p>',
                unsafe_allow_html=True
            )
    
    
    column_1, column_2 = st.columns(2)
    column_1.markdown(
                f'<p class="font-style" >Prediction </p>',
                unsafe_allow_html=True
            )
    
    prediction_str = prediction
    probabilities = prediction_proba
    
    dct = {0: 'Do not Have Heart Desease', 1: 'Have Heart Desease'}
    prediction_str = [dct[item] for item in prediction_str]  
    column_1.write(f"{prediction_str[0]}")
    
    column_2.markdown(
                '<p class="font-style" >Probability </p>',
                unsafe_allow_html=True
            )
    column_2.write(f"{probabilities[0][prediction[0]]}")
    
    fig = plot_pie_chart(probabilities)
    st.markdown(
                '<p class="font-style" >Probability Distribution</p>',
                unsafe_allow_html=True
            )
    st.plotly_chart(fig, use_container_width=True)  


elif page == "Compare Models":

    # Compare models.
    st.title("Compare Models: ")
    # dictionary with list object in values
    details = {
     'Algorithm Name' : ['Catboost', 'Voting Classifier'],
    'Accuracy' : [83, 84]
    }
    metric1 = pd.DataFrame(details)
    st.write(metric1.to_html(index=False), unsafe_allow_html=True)  
    #st.write(metric1)
    
    
    
    
    
    
    # st.subheader('Prediction')
    # churn_labels = np.array(['No','Yes'])
    # st.write(churn_labels[prediction])
    
    # st.subheader('Prediction Probability')
    # st.write(prediction_proba)
    
    
    
    
