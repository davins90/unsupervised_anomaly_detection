def main():
    import streamlit as st
    import pickle
    import requests
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import prince
    
    from modules import machine_learning_utils as mlu
    from io import BytesIO
    
    st.title("connect")
    
    file = "https://github.com/davins90/unsupervised_anomaly_detection/blob/master/src/data_lake/output/train_eng.pkl?raw=true"

    db = BytesIO(requests.get(file).content)
    df = pickle.load(db)
    
    df = df.astype('object')
    num_col = ["TransactionAmt","num_accounts_related_to_user","num_days_previous_transaction"]
    for i in df:
        if i in num_col:
            df[i] = df[i].astype('float')

    # 1.0 Pivot Analysis

    st.write(df.head())
    st.write(df.shape)

    x = st.selectbox("First element of conditiong: ",df.columns,index=2)
    y = st.selectbox("First element of conditiong: ",df.columns,index=6)

    num = pd.crosstab(index=df[y],columns=df[x],values=df[y],aggfunc='count').fillna(0.0)

    fig = px.bar(num)
    fig.show()
    st.plotly_chart(fig)

    # 2.0 MCA

    df2 = mlu.class_imbalance(df,'isFraud')
    df2['isFraud'] = df2['isFraud'].astype(str)
    cat = df2.select_dtypes(exclude='float64')
    cat = cat[['isFraud','ProductCD','DeviceType','browser_enc']]


    mca = mlu.compute_mca(prince.MCA,cat,3,5,True,True,'auto',2)
    fig = mlu.plot_coordinates_plotly2d(model=mca,X=cat,show_column_points=True)
    fig.show()
    st.plotly_chart(fig)

    # 3.0 Hist
    num = df.select_dtypes(include='float64')
    col = st.selectbox("Choose column to see its distribution",num.columns)

    fig = px.histogram(num[col],nbins=20)
    fig.show()
    st.plotly_chart(fig)
    
    