def main():
    import streamlit as st
    import gdown
    import pandas as pd
    import requests
    
    from modules import machine_learning_utils as mlu
    
    
    st.title("fraud bulk")
    
    # download from gdrive
    url = "https://drive.google.com/file/d/1av_NaF0lQqhQOJewFA0NgRzEl9M9Qfgv/view?usp=share_link"
    file_id=url.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    df = pd.read_pickle(dwn_url)
    
    st.write(df.head(1))
    
    # 2.0 features eng
    df = mlu.features_eng(df,'anomaly')
    
    # 3.0 model testing
    cols = ['card1', 'card2', 'card3','card5','M4', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
       'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_32', 'id_33', 'id_34',
       'id_35', 'id_36', 'id_37', 'id_38','num_transaction_per_time']
    
    df = df.drop(columns=cols)
    
    for i in df:
        if df[i].dtypes == 'object':
            df[i] = df[i].astype(str)
            
    X_test = df.drop(columns='isFraud')
    
    st.write(X_test.head(1))
    st.write(X_test.shape)

    
    if st.button("Submit"):
        ris = requests.post(f"http://fast_api:8000/predict_fraud_bulk/",json=X_test.to_dict())
        st.write(ris.json())