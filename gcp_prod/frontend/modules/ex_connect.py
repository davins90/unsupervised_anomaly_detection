def main():
    import streamlit as st
    import pickle
    import requests
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import prince
    
    
    from modules import machine_learning_utils as mlu
    from modules import knowledge_graph_utils as kg
    from modules import utils
    
    
    st.markdown("## Data Retrieval")
    
    df = utils.data_retrieval("https://drive.google.com/file/d/1axLbIYAxQbVnLQPNfEFfCyg_Eq5XSioi/view?usp=share_link")
    
    df = df.astype('object')
    num_col = ["TransactionAmt","num_accounts_related_to_user","num_days_previous_transaction"]
    for i in df:
        if i in num_col:
            df[i] = df[i].astype('float')
    
    st.write("Below is a sample of the user database that was connected to and its overall size.")
    st.write(df.sample(2))
    st.write(df.shape)

    # 1.0 Pivot Analysis
    st.markdown("## Pivot Analysis")
    st.write("This type of analysis allows us to query the database and filter the results for the columns we intend to analyze. For example, in case we wanted to extract the number of frauds for each product category we would simply select in the first box the 'isFraud' field and in the second box the 'ProductCD' field. Then a bar graph useful for graphical representation is generated.")
    
    form = st.form(key='my_form')
    x = form.selectbox("First element of conditiong: ",df.columns,index=0)
    y = form.selectbox("Second element of conditiong: ",df.columns,index=6)
    num = pd.crosstab(index=df[y],columns=df[x],values=df[y],aggfunc='count').fillna(0.0)
    submit_button = form.form_submit_button(label='Submit')

    fig = px.bar(num)
    fig.update_layout(title="Pivot Analysis",xaxis_title="{}".format(y) + " categories",yaxis_title="Counts")
    fig.show()
    st.plotly_chart(fig)

    # 2.0 MCA
    st.markdown("## MCA Analysis")
    st.write("Through the analysis of the multiple correspondences ([MCA](https://en.wikipedia.org/wiki/Multiple_correspondence_analysis)) it is possible to represent how the categories of data of a dataset behave, and to have in this way a summary view of what are the main relationships present. By search for 'isFraud' dot in the chart we can easily see what is the neighborhood of this field in this preliminary view.")
    
    df2 = mlu.class_imbalance(df,'isFraud')
    df2['isFraud'] = df2['isFraud'].astype(str)
    cat = df2.select_dtypes(exclude='float64')
    cat = cat[['isFraud','ProductCD','DeviceType','browser_enc']]


    mca = mlu.compute_mca(prince.MCA,cat,3,5,True,True,'auto',2)
    fig = mlu.plot_coordinates_plotly2d(model=mca,X=cat,show_column_points=True)
    fig.show()
    st.plotly_chart(fig)

    # 3.0 Hist
    st.markdown("## Distribution Analysis")
    st.write("The following is useful to represent the distribution of continuous variables in the database.")
    
    num = df.select_dtypes(include='float64')
    col = st.selectbox("Choose column to see its distribution",num.columns)

    fig = px.histogram(num[col],nbins=20)
    fig.show()
    st.plotly_chart(fig)
    
    # 4.0 Network Analysis
    st.markdown("## Network Analysis")
    st.write("In order to better study the relationships in the database, it was planned to construct a [knowledge graph](https://en.wikipedia.org/wiki/Knowledge_graph) in which the relationships between some of the major fields were constructed and colored to facilitate visualization. This approach, taking advantage of graph theory, can make it possible to look at a problem not only by analyzing the individual subjects under consideration but also the relationships between them and thus uncover new possible patterns otherwise hidden. \n Below is the color legend: \n - Nodes related to fraudulent transactions have been highlighted in red. \n - In yellow the device type (mobile/desktop). \n - In orange the browser type. \n - In blue the type of product. \n - In green the rest. \n Future developments of this approach may look to the generation of GNN models to make more accurate predictions and, with the appropriate data, better study the relationships.")
    
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
    
    