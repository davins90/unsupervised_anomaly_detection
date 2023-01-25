def main():
    import streamlit as st
    import pickle
    import requests
    
    from io import BytesIO
    
    st.title("connect")
    
    file = "https://github.com/davins90/unsupervised_anomaly_detection/blob/master/src/data_lake/output/train_eng.pkl?raw=true"

    db = BytesIO(requests.get(file).content)
    db = pickle.load(db)

    st.write(db.head(3))
    
    