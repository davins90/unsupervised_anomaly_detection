import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from pyvis import network as net
from pyvis.network import Network

# ##

def building_relation_dict(rels):
    """
    
    """
    dt = dict()
    for i in range(len(rels)):
        dt[i] = dict()
        dt[i]['source'] = rels[i][0]
        dt[i]['target'] = rels[i][1]
        dt[i]['rel'] = rels[i][2]
    return dt

def building_adj_list(dt,df):
    """
    
    """
    db = []
    for i in df.index:
        # select one row 
        row = df.loc[df.index==i]
        # create empty dataframe to be filled with info taken from the row selected
        gp = pd.DataFrame(columns=['source','target','rel'])
        # for loop for taking index and name of the dict of the relationship
        for y in dt.keys():
            for z in dt[y].keys():
                # for the rel value, we need to extract the relatioship value from the dict not from the row of the dataframe! 
                if z == 'rel':
                    gp.at[y,z] = dt[y][z]
                else:
                    gp.at[y,z] = row[dt[y][z]][i]
        db.append(gp)
    final = pd.concat(db)
    final = final.reset_index().drop(columns='index')
    final = final.drop_duplicates()
    return final

def building_network(final,final2,df):
    """
    
    """
    net = Network(directed=True,height='900px', width='100%',notebook=False,cdn_resources='in_line',select_menu=True, filter_menu=True)
    nodes = list(pd.unique(final[['source','target']].values.ravel('K')))
    nodes2 = list(pd.unique(final2[['source','target']].values.ravel('K')))
    tra = list(df['TransactionID'].unique())
    pdr = list(df['ProductCD'].unique())
    bro = list(df['browser_enc'].unique())
    devt = list(df['DeviceType'].unique())
    devi = list(df['device_info_v4'].unique())
    # add nodes
    for e in nodes:
        if e in pdr:
            net.add_node(e,color='blue')
        elif e in bro:
            net.add_node(e,color='orange')
        elif e in devt:
            net.add_node(e,color='yellow')
        elif e in devi:
            net.add_node(e,color='gray')
        elif e in tra:
            val = df[df['TransactionID']==e]['TransactionAmt'].values[0]
            if e in nodes2:
                net.add_node(e,color='red',value=val)
            else:
                net.add_node(e,color='green',value=val)
        elif e in nodes2:
            net.add_node(e,color='red')
            
        else:
            net.add_node(e,color='green')
    # add edges
    for i in final.index:
        row = final.loc[final.index==i]
        net.add_edge(row['source'].item(),row['target'].item(),title=row['rel'].item(),label=row['rel'].item())
#     net.show_buttons(filter_=['physics'])
    net.set_options("""const options = {
  "physics": {
    "repulsion": {
      "springLength": 400,
      "springConstant": 0.1,
      "nodeDistance": 400
    },
    "minVelocity": 0.75,
    "solver": "repulsion"
  }
}""")
    # save in cache
    try:
        path = '/tmp'
        net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
    except:
        HtmlFile = open(f'app/modules/kg_v29_11.html', 'r', encoding='utf-8')
    return components.html(HtmlFile.read(), height=900)
