
import streamlit as st
import pycaret.regression as re
import pycaret.classification as cl
import os
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report



with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Auto ML")
    choice = st.radio("Navigation", ["Upload", "Profile", "Model","Predict", "Download"])
    st.info("We use PyCaret for Modelling")

if os.path.exists("dataset.csv"): 
    df = pd.read_csv('dataset.csv', index_col=None)

try:
    if choice == "Upload": 
        st.title("Upload")
        file = st.file_uploader("Upload Your Dataset")
        if file: 
            df = pd.read_csv(file, index_col=None)
            df.to_csv("dataset.csv", index=None)
            st.dataframe(df)
except NameError:
    None

if choice == "Profile": 
    st.title("Profile")
    if st.button("EDA"):
        profile_df = df.profile_report()
        st_profile_report(profile_df)


if choice=="Model":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    chosen_id = st.selectbox("Choose ID for prediction",df.columns)
    choice_2=st.radio('Select which Machine learning should be used',['Regression','Classification']) 

    if choice_2=="Regression":
        if st.button("start"):
            re.setup(df, target=chosen_target, silent=True,normalize=True,transformation=True)
            setup_df = re.pull()
            st.dataframe(setup_df)
            best_model = re.compare_models()
            compare_df = re.pull()
            st.dataframe(compare_df)
            st.plotly_chart(re.plot_model(best_model,"residuals"))
            st.plotly_chart(re.plot_model(best_model,"learning"))
            model = re.finalize_model(best_model)
            
            st.write("Modelling Finished")
            st.write("Upload test data")
            try:
                file_ = st.file_uploader("Upload Your test Dataset")
                if file_: 
                    df_test = pd.read_csv(file_, index_col=None)
                    df_test.to_csv("dataset.csv", index=None)
                    st.dataframe(df_test)
            except NameError:
                None

    if choice_2=="Classification":
        if st.button("start"):
            cl.setup(df, target=chosen_target, silent=True,normalize=True,transformation=True)
            setup_df = cl.pull()
            st.dataframe(setup_df)
            best_model = cl.compare_models()
            compare_df = cl.pull()
            st.dataframe(compare_df)
            st.plotly_chart(cl.plot_model(best_model,"residuals"))
            st.plotly_chart(cl.plot_model(best_model,"learning"))
            model_ = cl.finalize_model(best_model)
            
            st.write("Modelling Finished")
            st.write("Upload test data")
            try:
                file__ = st.file_uploader("Upload Your test Dataset")
                if file__: 
                    df_test = pd.read_csv(file__, index_col=None)
                    df_test.to_csv("dataset.csv", index=None)
                    st.dataframe(df_test)
            except NameError:
                None
   

if choice=='Predict':
    if st.button("predict regression"):    
        predictions = re.predict_model(model,data=df_test)
        pre_data=pd.DataFrame({
                    chosen_id:predictions[chosen_target]
                })
    if st.button("predict classification"):    
        predictions = cl.predict_model(model_,data=df_test)
        pre_data=pd.DataFrame({
                    chosen_id:predictions[chosen_target]
                })

if choice=="Download":
     with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")











