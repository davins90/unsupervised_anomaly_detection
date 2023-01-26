from fastapi import FastAPI

import pickle

app = FastAPI()

with open("api/trained_model_v2.pkl","rb") as m:
    model = pickle.load(m)

@app.get('/')
def get_root():
    return {'message': 'Welcome to the fraud detection & clustering API'}

@app.post("/predict")
async def predict(data):
    data = pd.Dataframe.from_dict(data)
    prediction = model.predict_proba(data)[:,1]
    return {"prediction: ",prediction}
