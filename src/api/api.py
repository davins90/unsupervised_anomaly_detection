# from fastapi import FastAPI
# from typing import Union
# from pydantic import BaseModel

# app = FastAPI()

# class Item(BaseModel):
#     name: str
#     price: float
#     is_offer: Union[bool, None] = None

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}

from fastapi import FastAPI, Request

app = FastAPI()

# @app.get("/sum/{a}/{b}")
# async def read_item(a: int, b: int):
#     return {"sum": a + b}

@app.get('/')
def get_root():
	return {'message': 'Welcome to the spam detection API'}

def edit(x,y):
    x = x*2
    y = y*2
    return x,y

@app.post("/sum")
async def read_item(request: Request):
    data = await request.json()
    a = data["a"]
    b = data["b"]
    a,b = edit(a,b)
    return {"sum": a + b}
