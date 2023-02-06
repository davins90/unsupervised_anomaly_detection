def main():
    import streamlit as st
    import requests
    import pandas as pd
    
    from modules import machine_learning_utils as mlu
    from modules import utils
    
    st.title("fraud single")
    
    df = utils.data_retrieval("https://drive.google.com/file/d/1axLbIYAxQbVnLQPNfEFfCyg_Eq5XSioi/view?usp=share_link")
    
    cols = ['card1', 'card2', 'card3','card5','M4', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
       'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_32', 'id_33', 'id_34',
       'id_35', 'id_36', 'id_37', 'id_38','num_transaction_per_time']
    
    df = df.drop(columns=cols)
    
    num_col = ["TransactionAmt","num_accounts_related_to_user","num_days_previous_transaction"]
    
    df = df.astype('object')
    for i in df:
        if i in num_col:
            df[i] = df[i].astype(float)
    
    features = dict()
    

    for i,j in enumerate(df):
        if j == "isFraud":
            pass
        elif j in num_col:
            features[j] = st.number_input('Insert number for {} related to the transaction under examination: '.format(j),key=i)
        else:
            features[j] = st.selectbox("Select type of {} related to the transaction: ".format(j),df[j].unique(),key=i)
            
    
    if st.button("Submit"):
        ris_ml = requests.post(f"https://backend-4b-ylpi3mxsaq-oc.a.run.app/predict_fraud_ml",json=features).json()
        ris_hk = requests.post(f"https://backend-4b-ylpi3mxsaq-oc.a.run.app/predict_fraud_hk",json=features).json()
        st.write("ris ml: ",ris_ml)
        st.write("ris hk: ",ris_hk)
        final_score = mlu.beta_fusion(ris_ml,ris_hk,0.6)
        st.write(final_score)