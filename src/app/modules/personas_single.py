def main():
    import streamlit as st
    import pickle
    import requests
    import pandas as pd
    
    from modules import machine_learning_utils as mlu
    from io import BytesIO
    
    st.title("pers single")
    
    with open("app/modules/models/clustering_imputation_cat.pkl","rb") as imc:
        imputation_cat = pickle.load(imc)

    with open("app/modules/models/clustering_imputation_num.pkl","rb") as imn:
        imputation_num = pickle.load(imn)

    with open("app/modules/models/clustering_scaler_num.pkl","rb") as sn:
        scaler_num = pickle.load(sn)
    
    file = "https://github.com/davins90/unsupervised_anomaly_detection/blob/master/src/data_lake/output_prod/train.pkl?raw=true"
    db = BytesIO(requests.get(file).content)
    df = pickle.load(db)
    
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
            features[j] = st.number_input('Insert number for {} related to the transaction under examination: '.format(j),key=i)
        else:
            features[j] = st.selectbox("Select type of {} related to the transaction: ".format(j),df[j].unique(),key=i)
            
    user = dict()
    for i,j in features.items():
        user[i] = [j]
    us = pd.DataFrame(user)
    
    us = mlu.clustering_preparation(us,'prediction',imputation_num,scaler_num,imputation_cat)
    
    us = mlu.clustering_encoding(us,'prediction')
    
    st.write(us.to_dict())
    
    if st.button("Submit"):
        result = requests.post(f"http://fast_api:8000/predict_personas/",json=us.to_dict()).json()
        st.write(result)
    
    
    
    