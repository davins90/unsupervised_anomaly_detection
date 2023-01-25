def main():
    import streamlit as st
    import pickle
    import requests
    import pandas as pd
    
    from modules import machine_learning_utils as mlu
    from io import BytesIO
    
    st.title("fraud single")
    file = "https://github.com/davins90/unsupervised_anomaly_detection/blob/master/src/data_lake/output/train_eng.pkl?raw=true"

    db = BytesIO(requests.get(file).content)
    df = pickle.load(db)
    
    transaction_amt = st.number_input('Insert amount of the transacation under examination: ')
    num_accounts_realted = st.number_input('Insert number of accounts related to the user: ')
    days_since_previous = st.number_input('Insert the number of days since last transaction: ')
    
    num_col = ["TransactionAmt","num_accounts_related_to_user","num_days_previous_transaction"]
    
    for i,j in enumerate(df):
        if j in num_col:
            pass
        else:
            print(j)
            j = st.selectbox("Select type of {} related to the transaction: ".format(j),df[j].unique(),key=i)
    # product = st.selectbox("Select type of product related to the transaction: ",df['ProductCD'].unique())
    