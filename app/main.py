import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from set_up_model import TtsModel
from fastapi.responses import FileResponse

app = FastAPI(title="Predicting Wine Class")


class Text(BaseModel):
    input : str


class ResponseAudi(BaseModel):
    output : list

@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    with open("tacotron.pkl", "rb") as file:
        global clf
        clf = pickle.load(file)


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:80/docs"


@app.post("/predict", response_class=FileResponse)
def predict(text: Text):

    pred = clf.predict(text.input)
    # pred = (pred.cpu().detach().numpy()[0]).tolist()
    # return {"Prediction": pred}
    return pred
