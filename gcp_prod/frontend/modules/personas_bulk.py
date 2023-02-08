def main():
    import streamlit as st
    import pandas as pd
    import requests
    import plotly.express as px
    
    from modules import machine_learning_utils as mlu
    from modules import utils
    
    st.markdown("## Bulk Personas Prediction")
    st.write("On this page, at the current state, we are automatically connected to the database and the prediction of the cluster to which each customer belongs is computed with a dedicated API.Below is a sample of the DB data. Following there are the description of each cluster (personas).")
    
    # download from gdrive
    
    df = utils.data_retrieval("https://drive.google.com/file/d/1av_NaF0lQqhQOJewFA0NgRzEl9M9Qfgv/view?usp=share_link")
    
    st.write(df.head(2))
    
    st.markdown("## Clusters Personas Description")
    st.write("#### Cluster 0: Regular Desktop User with Credit Card")
    st.write("This cluster represents a user who regularly makes transactions using a desktop computer with the Chrome browser and Windows operating system. They have a moderate number of connected accounts and have a moderate time interval between transactions. This user utilizes a VISA credit card for purchasing type 'C' products.")
    st.write("#### Cluster 1: Occasional High-Value Desktop User with Debit Card")
    st.write("This cluster represents a user who occasionally makes high-value transactions using a desktop computer with the Chrome browser and Windows operating system. They have a moderate number of connected accounts and a high time interval between transactions. This user utilizes a VISA debit card for purchasing type 'R' products.")
    st.write("#### Cluster 2: Frequent Low-Value Mobile User with Debit Card")
    st.write("This cluster represents a user who frequently makes low-value transactions using a mobile device with the Chrome browser and an 'OTHER' operating system. They have a high number of connected accounts and a moderate time interval between transactions. This user utilizes a VISA debit card for purchasing type 'C' products. **This seems to be the cluster in which the most 'mysterious' customers are present, among whom we find some of the possible fraudsters**.")
    st.write("#### Cluster 3: Occasional High-Value Mobile User with Credit Card")
    st.write("This cluster represents a user who occasionally makes high-value transactions using a mobile device with the Safari browser and IOS operating system. They have a low number of connected accounts and a low time interval between transactions. This user utilizes a VISA credit card for purchasing type 'H' products.")
    
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
    
    st.write("Below a sample of the data prepared and engineered, ready for the prediction. ")
    st.write(df.head(2))

    if st.button("Submit"):
        ris = requests.post(f"https://backend-4b-ylpi3mxsaq-oc.a.run.app/predict_personas_bulk",json=df.to_dict())
        risu = pd.DataFrame(ris.json())
        sub = risu.filter(regex='cluster_labels_pred',axis=1)
        st.write("Once the prediction is completed, here is the distribution of the three calculated results below.")
        sub2 = sub['cluster_labels_pred'].value_counts().reset_index()
        sub2 = sub2.rename(columns={'index':'cluster_predicted','cluster_labels_pred':'count'})
        
        fig = px.bar(sub2,x='cluster_predicted',y='count')
        fig.show()
        st.plotly_chart(fig)
    