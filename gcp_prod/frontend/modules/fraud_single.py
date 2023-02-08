def main():
    import streamlit as st
    import requests
    import pandas as pd
    
    from modules import machine_learning_utils as mlu
    from modules import utils
    
    st.markdown("## Single Transaction prediction")
    st.write("By filling out the following form with data from the transactions under consideration, three different fraud scores can be obtained: \n - the first one as the output of the processed machine-learning model; \n - the second as the output of a rule-based model, designed based on the investigators' knowledge; \n - the third, final, is a Bayesian-averaged score of the previous two.")
    
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
            
    st.markdown("### Input form")
    form = st.form(key='my_form')
    features = dict()
    for i,j in enumerate(df):
        if j == "isFraud":
            pass
        elif j in num_col:
            features[j] = form.number_input('Insert number for {} related to the transaction under examination: '.format(j),key=i)
        else:
            features[j] = form.selectbox("Select type of {} related to the transaction: ".format(j),df[j].unique(),key=i)
    
    submit_button = form.form_submit_button(label='Submit')

    if submit_button is True:
        ris_ml = requests.post(f"https://backend-4b-ylpi3mxsaq-oc.a.run.app/predict_fraud_ml",json=features).json()
        ris_hk = requests.post(f"https://backend-4b-ylpi3mxsaq-oc.a.run.app/predict_fraud_hk",json=features).json()
        final_score = mlu.beta_fusion(ris_ml,ris_hk,0.6)

        st.markdown("### Output")
        
        st.write("Score from the Machine Learning tool: ")
        utils.gauge_chart(ris_ml)
        
        st.write("Score from the rule based tool: ")
        utils.gauge_chart(ris_hk)
        
        st.write("Final score: ")
        utils.gauge_chart(final_score)
