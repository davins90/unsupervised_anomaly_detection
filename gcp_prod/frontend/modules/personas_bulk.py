def main():
    import streamlit as st
    import pandas as pd
    import requests
    
    from modules import machine_learning_utils as mlu
    from modules import utils
    
    st.title("perso bulk")
    
    # download from gdrive
    
    df = utils.data_retrieval("https://drive.google.com/file/d/1av_NaF0lQqhQOJewFA0NgRzEl9M9Qfgv/view?usp=share_link")
    
    st.write(df.head(1))
    
    df = mlu.features_eng(df,'clustering')
    
    df = df[df['isFraud']==0]
    df = df.drop(columns='isFraud')
    
    cols = ['card1', 'card2', 'card3', 'card5', 'M4', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
       'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_32', 'id_33', 'id_34',
       'id_35', 'id_36', 'id_37', 'id_38','num_transaction_per_time','addr1','P_emaildomain','R_emaildomain','addr2','multi_transaction_per_time']
    
    df = df.drop(columns=cols)
    
    num_col = ["TransactionAmt","num_accounts_related_to_user","num_days_previous_transaction"]
    
    df = df.astype(str)
    for i in df:
        if i in num_col:
            df[i] = df[i].astype(float)
            
    st.write(df.head(1))
    
    if st.button("Submit"):
        ris = requests.post(f"http://fast_api:8000/predict_personas_bulk/",json=df.to_dict())
        st.write(ris.json())