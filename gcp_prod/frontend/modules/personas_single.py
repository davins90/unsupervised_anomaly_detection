def main():
    import streamlit as st
    import requests
    import pandas as pd
    
    from modules import machine_learning_utils as mlu
    from modules import utils
    
    st.title("pers single")

    df = utils.data_retrieval("https://drive.google.com/file/d/1av3XEv59qOykE8v7tDu6GBJFbIkABiC8/view?usp=share_link")
    
    df = mlu.features_eng(df,'clustering')
    
    df = df[df['isFraud']==0]
    df = df.drop(columns='isFraud')
    
    cols = ['card1', 'card2', 'card3', 'card5', 'M4', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
       'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_32', 'id_33', 'id_34',
       'id_35', 'id_36', 'id_37', 'id_38','num_transaction_per_time','addr1','P_emaildomain','R_emaildomain','addr2','multi_transaction_per_time']
    
    df = df.drop(columns=cols)
    
    st.write(df.head(1))
    
    num_col = ["TransactionAmt","num_accounts_related_to_user","num_days_previous_transaction"]
    
    df = df.astype(str)
    for i in df:
        if i in num_col:
            df[i] = df[i].astype(float)
    
    features = dict()

    for i,j in enumerate(df):
        if j == "customer_id":
            pass
        elif j in num_col:
            features[j] = st.number_input('Insert number for {} related to the transaction under examination: '.format(j),key=i,value=0.028)
        else:
            features[j] = st.selectbox("Select type of {} related to the transaction: ".format(j),df[j].unique(),key=i)
            
    user = dict()
    for i,j in features.items():
        user[i] = [j]
    us = pd.DataFrame(user)
    
    st.write(us.to_dict())
    
    if st.button("Submit"):
        response = requests.post("https://backend-4b-ylpi3mxsaq-oc.a.run.app/predict_personas/",json=us.to_dict())
        if response.status_code == 200:
            result = response.json()
            st.write("Prediction:", result)
        else:
            st.write("Error in API call: ",response)
    
    
    
    