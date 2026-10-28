# To convert the model into api -> this file is used(api code)
## We need fastapi->rest api, uvicorn-8000, basemodel - structure, @ - annotation-> set of instructions, "/""" -  routers

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("linearmodel.pkl")

class datascience(BaseModel):
    StudyHours : float

# api end point

@app.get("/")
def test():
    return {"msg":"uvicorn is running"}

# Data insert Studyhrs = marks api
@app.post("/predict")
def datainsert(ds:datascience):
    marks = model.predict([[ds.StudyHours]])
    return {"predictedmarks":round(float(marks[0]),2)}

# How to run this code in terminal:
# python -m uvicorn main:app --reload
# It will run at -> http://127.0.0.1:8000
# Interactive API documentation / testing -> http://127.0.0.1:8000/docs
