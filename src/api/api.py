from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from kmodes.kprototypes import KPrototypes

import pickle
import pandas as pd
import numpy as np

app = FastAPI()

# Load Model

with open("api/trained_model_v2.pkl","rb") as m:
    model = pickle.load(m)
    
with open("api/clustering_model_v1.pkl","rb") as cm:
    clus = pickle.load(cm)
    
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

        
@app.get('/')
def get_root():
    return {'message': 'Welcome to the fraud detection & clustering API'}
    
    
@app.post("/predict_fraud")
async def predict_fraud(data: fraud_prediction):
    data = jsonable_encoder(data)
    for key, value in data.items():
        data[key] = [value]
    single_instance = pd.DataFrame.from_dict(data)
    prediction = model.predict_proba(single_instance)[:,1]
    return prediction[0]

@app.post("/predict_personas")
async def predict_personas(data: personas, request: Request):
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    ris = clus.predict(X=df[['TransactionAmt', 'num_accounts_related_to_user', 'num_days_previous_transaction',
                             'product_enc', 'card4_enc', 'card6_enc', 'DeviceType_enc','browser_enc2','device_info_v4_enc']],categorical=[3,4,5,6,7,8])[0]
    ris = np.float64(ris)
    return ris

