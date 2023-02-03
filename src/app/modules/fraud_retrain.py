def main():
    import sys
    sys.path.append('../../../')
    
    import streamlit as st
    import pickle
    import pandas as pd

    from ploomber_engine import execute_notebook

    
    st.title("fraud retrain")
    
    with open("data_lake/output_prod/df_under.pkl","rb") as m:
        df = pd.read_pickle(m)
        
    st.write(df.head(1))
    
    
    st.write("Start")
    
    train_dim = st.number_input("sele",step=0.5,value=0.7)
    val_test_dim = 0.15
    

    out = execute_notebook("notebooks/dev_version/2_data_preparation/ploomber_test.ipynb",
                           "notebooks/dev_version/2_data_preparation/ploomber_test2.ipynb",log_output=True,verbose=True,
                           parameters={"train_dim":train_dim,
                                       "val_test_dim":val_test_dim})

    st.write("end")

