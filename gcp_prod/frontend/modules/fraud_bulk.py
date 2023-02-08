def main():
    import streamlit as st
    import pandas as pd
    import requests
    import numpy as np
    import plotly.express as px
    
    from modules import machine_learning_utils as mlu
    from modules import utils
    
    st.markdown("## Bulk Transaction Prediction")
    st.write("On this page, at the current state, we are automatically connected to the database and the computation of the fraud probability score is performed on all transactions provided.Below is a sample of the DB data.")
    
    # download from gdrive
    
    df = utils.data_retrieval("https://drive.google.com/file/d/1av_NaF0lQqhQOJewFA0NgRzEl9M9Qfgv/view?usp=share_link")
    
    st.write(df.head(2))
    
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
    
    st.write("Below a sample of the data prepared and engineered, ready for the prediction. ")
    st.write(X_test.head(2))
    st.write(X_test.shape)

    if st.button("Submit"):
        ris = requests.post(f"https://backend-4b-ylpi3mxsaq-oc.a.run.app/predict_fraud_bulk",json=X_test.to_dict())
        risu = pd.DataFrame(ris.json())
        sub = risu.filter(regex='score',axis=1)
        st.write("Once the prediction is completed, here is the distribution of the three calculated results below.")
        sub = sub.rename(columns={'probability_score':'machine_learning_score',
                        'warning_score':'rule_based_score'})
        
        sub2 =pd.DataFrame(dict(series=np.concatenate((["machine_learning_score"]*sub.shape[0],["rule_based_score"]*sub.shape[0],["final_score"]*sub.shape[0])), 
                              data=np.concatenate((sub['machine_learning_score'],sub['rule_based_score'],sub['final_score']))))

        fig = px.histogram(sub2, x="data", color="series", barmode="overlay")
        fig.show()
        st.plotly_chart(fig)
