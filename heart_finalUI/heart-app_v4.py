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

Please select Menu""", ["Main Page", "Tables", "Compare Models"])

#-------------------------------------------------


if page == "Main Page":

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
    # Cardiac Desease Prediction
    """)
    
    df_selected = pd.read_csv(r'C:\Users\sndip\Downloads\streamlit_heart_final\streamlit_heart\heart_finalUI\final_heart_data_v1.csv')
    df_selected_all = df_selected.copy()
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="heart_prediction_data.csv">Download CSV File</a>'
        return href
    
    # Reads the saved classification model
    load_clf = pickle.load(open('./heart_clf.pkl', 'rb'))

    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown(filedownload(df_selected_all), unsafe_allow_html=True)
    
    
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        input_df=input_df.drop('condition',axis=1)
        # Apply model to make predictions
        prediction = load_clf.predict(input_df)
        prediction = pd.DataFrame(prediction,columns =['Prediction']).reset_index(drop=True)
        upl_df = pd.concat([prediction,input_df],axis=1)
        st.write(upl_df)
    else:
        def user_input_features():
            age = st.sidebar.slider('Age', 1.0,80.0, 1.0
                                    )
            sex = st.sidebar.selectbox('Sex',(1,0))
            cp = st.sidebar.selectbox('Chest pain',(1.0, 2.0, 3.0, 4.0))
            trestbps = st.sidebar.slider('Resting BP', 0.0,200.0, 20.0)
            chol = st.sidebar.slider('Serum cholesterol', 0.0,605.0, 100.0)
            fbs = st.sidebar.selectbox('Fasting blood sugar>120 mg/dl',(0.0,1.0))
            restecg = st.sidebar.selectbox('Resting ecg',(0.0,1.0,2.0))
            thalach = st.sidebar.slider('Maximum heart rate', 60.0,205.0, 70.0)
            exang = st.sidebar.selectbox('Exercise induced angina',(0.0,1.0))
            oldpeak = st.sidebar.slider('oldpeak', -2.6,6.5, 1.0)
            slope = st.sidebar.selectbox('slope',(1.0,2.0,3.0))
    
            data = {'Age':[age], 
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
    

        df_data = pd.read_csv(r'C:\Users\sndip\Downloads\streamlit_heart_final\streamlit_heart\heart_finalUI\final_heart_data_v1.csv')
        df_data.fillna(0, inplace=True)
        data_df = df_data.drop(columns=['condition'])
        df = pd.concat([input_df,data_df],axis=0)
        df = df[:1] # Selects only the first row (the user input data)
        df.fillna(0, inplace=True)


        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope']
        df = df[features]


        # Reads in saved classification model
        load_clf = pickle.load(open('./heart_clf.pkl', 'rb'))

# Apply model to make predictions
        prediction = load_clf.predict(df)
        prediction_proba = load_clf.predict_proba(df)
# Displays the user input features
        st.subheader('User Input features')
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        st.write(df)
        import plotly.graph_objects as go

        def plot_pie_chart(probabilities):
                fig = go.Figure(
                    data=[go.Pie(
                            labels=list(set(df_data.condition)),
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
    

elif page == "Tables":


    st.title("Approach 1- voting classifier with each group")
    metric1 = pd.read_csv(r"C:\Users\sndip\Downloads\streamlit_heart_final\streamlit_heart\heart_finalUI\Approach_1_acc_results.csv",index_col=0)
    st.write(metric1.to_html(index=False), unsafe_allow_html=True) 

    st.title("Approach 2- All Models")
    metric2 = pd.read_csv(r"C:\Users\sndip\Downloads\streamlit_heart_final\streamlit_heart\heart_finalUI\Approach2_all_models.csv")
    st.write(metric2.to_html(index=False), unsafe_allow_html=True) 
 
    st.title("Grid Search - Hyper parameter tuning Results")
    
    metric3 = pd.read_csv(r"C:\Users\sndip\Downloads\streamlit_heart_final\streamlit_heart\heart_finalUI\grid_results.csv",index_col=0)
    st.write(metric3.to_html(index=False), unsafe_allow_html=True) 
    

elif page == "Compare Models":

    # Compare models.
    st.title("With baseline parameters")
    
    metric1 = pd.read_csv(r"C:\Users\sndip\Downloads\streamlit_heart_final\streamlit_heart\heart_finalUI\Baseline_parameters.csv")
    st.write(metric1.to_html(index=False), unsafe_allow_html=True) 
    
    
    st.title("After parameter tuning")
    metric2 = pd.read_csv(r"C:\Users\sndip\Downloads\streamlit_heart_final\streamlit_heart\heart_finalUI\After_parameter_tuning.csv")
    st.write(metric2.to_html(index=False), unsafe_allow_html=True)