def main():
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import prince
    import requests
    import pickle
    
    from modules import machine_learning_utils as mlu
    from modules import knowledge_graph_utils as kg
    from modules import utils
    
    st.markdown("## Data Retrieval")
    file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx", "pkl"])

    if file is not None:
        st.write("File uploaded: ", file.name)
        if ".csv" in file.name:
            df = pd.read_csv(file)
            df = df.drop(columns=['Unnamed: 0'])
            df = mlu.features_eng(df,'anomaly')

        else:
            df = pd.read_pickle(file)
            df = mlu.features_eng(df,'anomaly')
            
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
        
        # 4.0 Network Analysis
        
        # download from gdrive
        
        df2 = utils.data_retrieval("https://drive.google.com/file/d/1as5pfJ3FuHnJ18BOh1-YRI6J9wemWDdn/view?usp=share_link")
        
        df2 = mlu.features_eng(df2,'network')
        df2 = df2[['customer_id','TransactionID','TransactionAmt','DeviceType','device_info_v4','browser_enc','ProductCD','isFraud']]
        
        df2 = df2.dropna(axis=0)
        df2 = df2.sample(frac=0.0005,random_state=4)
        
        for i in df2:
            if i != 'TransactionAmt':
                df2[i] = df2[i].astype(str)
        
        rel1 = ['customer_id','TransactionID','has done']
        rel2 = ['TransactionID','ProductCD','buying']
        rel3 = ['ProductCD','DeviceType','by']
        rel4 = ['TransactionID','DeviceType','with']
        rel5 = ['TransactionID','browser_enc','on']
        
        rels = [rel1,rel2,rel3,rel4,rel5]
        
        dt = kg.building_relation_dict(rels)
        
        final = kg.building_adj_list(dt,df2)
        
        # extract only fraud
        df3 = df2[df2['isFraud']=='1']
        
        rel1 = ['customer_id','TransactionID','has done']
        rels = [rel1]
        
        dt2 = kg.building_relation_dict(rels)
        
        final2 = kg.building_adj_list(dt2,df3)
        
        kg.building_network(final,final2,df2)
    else:
        st.warning("Still necessary build sample file for testing. Doing random sample of original dataframe after merging. Refactoring also of the wording of this pages according to the connect page.")
        