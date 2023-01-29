from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from kmodes.kprototypes import KPrototypes

import pickle
import pandas as pd

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
    ProductCD: object
    card4: object
    card6: object
    DeviceType: object
    browser_enc: object
    device_info_v4: object

        
@app.get('/')
def get_root():
    return {'message': 'Welcome to the fraud detection & clustering API'}
    
    
@app.post("/predict_fraud")
async def predict_fraud(data: fraud_prediction):
    data = jsonable_encoder(data)
    for key, value in data.items():
        data[key] = [value]
     # answer_dict = {k:[v] for (k,v) in jsonable_encoder(answer).items()}
    single_instance = pd.DataFrame.from_dict(data)
    prediction = model.predict_proba(single_instance)[:,1]
    return prediction[0]

@app.post("/predict_personas")
async def predict_personas(data: personas):
    data = jsonable_encoder(data)
    for key, value in data.items():
        data[key] = [value]
     # answer_dict = {k:[v] for (k,v) in jsonable_encoder(answer).items()}
    single_instance = pd.DataFrame.from_dict(data)
#     prediction = clus.predict(X=single_instance,categorical=[3,4,5,6,7,8])
    return single_instance

