def main():
    import streamlit as st
    import pickle
    import pandas as pd

    from ploomber_engine import execute_notebook

    st.title("Retraining Fraud Detection model")
    
    st.write("Start Data Preparation")
    
    train_dim = st.number_input("Select dimension of the training set (%)",step=0.5,value=0.7)
    val_test_dim = 0.15
    

    out = execute_notebook("notebooks/input/2.0_data_preparation_input.ipynb",None,log_output=True,verbose=True,
                           parameters={"train_dim":train_dim,
                                       "val_test_dim":val_test_dim})

    st.write("End Data Preparation")
    
    st.write("Start Features Engineering")
    
    out2 = execute_notebook("notebooks/input/3.0_features_eng_input.ipynb",None,log_output=True,verbose=True)
    
    st.write("End Features Enginnering")
    
    
    st.write("Start Traininig model")
    
    out3 = execute_notebook("notebooks/input/4.0_training_evaluation_input.ipynb",None,log_output=True,verbose=True)
    
    st.write("End Training model")
    

