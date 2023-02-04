from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from kmodes.kprototypes import KPrototypes
from typing import List

import pickle
import pandas as pd
import numpy as np
from api import machine_learning_utils as mlu

app = FastAPI()

# Load Model

with open("api/trained_model_v2.pkl","rb") as m:
    model = pickle.load(m)
    
with open("api/clustering_model_v1.pkl","rb") as cm:
    clus = pickle.load(cm)
    
# Load scalers and imputations

with open("api/clustering_imputation_cat.pkl","rb") as imc:
    imputation_cat = pickle.load(imc)

with open("api/clustering_imputation_num.pkl","rb") as imn:
    imputation_num = pickle.load(imn)

with open("api/clustering_scaler_num.pkl","rb") as sn:
    scaler_num = pickle.load(sn)

with open("api/log_scaler_bi.pkl","rb") as ls:
    scaler_bi = pickle.load(ls)
    
# Methods
class fraud_prediction(BaseModel):
    TransactionAmt: float
    ProductCD: object
    card4: object
    card6: object
    addr1: object
    addr2: object
    P_emaildomain: object
    R_emaildomain: object
    DeviceType: object
    num_accounts_related_to_user: float
    num_days_previous_transaction: float
    multi_transaction_per_time: object
    browser_enc: object
    device_info_v4: object
        
class personas(BaseModel):
    TransactionAmt: float
    num_accounts_related_to_user: float
    num_days_previous_transaction: float
    product_enc: int
    card4_enc: int
    card6_enc: int
    DeviceType_enc: int
    browser_enc2: int
    device_info_v4_enc: int

# Home

@app.get('/')
def get_root():
    return {'message': 'Welcome to the fraud detection & clustering API'}
    
# Fraud detection
    
@app.post("/predict_fraud_ml")
async def predict_fraud_ml(data: fraud_prediction):
    data = jsonable_encoder(data)
    for key, value in data.items():
        data[key] = [value]
    single_instance = pd.DataFrame.from_dict(data)
    prediction = model.predict_proba(single_instance)[:,1]
    return prediction[0]

@app.post("/predict_fraud_hk")
async def predict_fraud_hk(data: fraud_prediction):
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    df = mlu.log_beta_transform(df,scaler_bi,'num_accounts_related_to_user','num_days_previous_transaction')
    df = mlu.warning_score_prod(df,0.8)
    return df['warning_score'][0]


@app.post("/predict_fraud_bulk")
async def predict_fraud_bulk(data: dict):
    df = pd.DataFrame.from_dict(data)
    print(df.shape)
    # ml
    df['probability_score'] = model.predict_proba(df)[:,1]
    # hk
    df = mlu.log_beta_transform(df,scaler_bi,'num_accounts_related_to_user','num_days_previous_transaction')
    df = mlu.warning_score_prod(df,0.8)
    # final score
    df['final_score'] = df.apply(lambda x: mlu.beta_fusion(x['probability_score'],x['warning_score'],0.6),axis=1)
    df = df.fillna(0.0)
    print(df)
    return df.to_dict(orient='records')



# Clustering

@app.post("/predict_personas")
# async def predict_personas(data: personas, request: Request):
async def predict_personas(data: dict, request: Request):
    print(data)
#     df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    df = pd.DataFrame.from_dict(data)
    df = mlu.clustering_preparation(df,'prediction',imputation_num,scaler_num,imputation_cat)
    df = mlu.clustering_encoding(df,'prediction')
    ris = clus.predict(X=df[['TransactionAmt', 'num_accounts_related_to_user', 'num_days_previous_transaction',
                             'product_enc', 'card4_enc', 'card6_enc', 'DeviceType_enc','browser_enc2','device_info_v4_enc']],categorical=[3,4,5,6,7,8])[0]
    ris = np.float64(ris)
    return ris

@app.post("/predict_personas_bulk")
async def predict_personas_bulk(data: dict, request: Request):
    df = pd.DataFrame.from_dict(data)
    df = mlu.clustering_preparation(df,'prediction',imputation_num,scaler_num,imputation_cat)
    df = mlu.clustering_encoding(df,'prediction')
    df['cluster_labels_pred'] = clus.predict(X=df[['TransactionAmt', 'num_accounts_related_to_user', 'num_days_previous_transaction',
                             'product_enc', 'card4_enc', 'card6_enc', 'DeviceType_enc','browser_enc2','device_info_v4_enc']],categorical=[3,4,5,6,7,8])
    df = df.fillna(0.0)
    return df.to_dict(orient='records')

