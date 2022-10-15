from pathlib import Path
import pickle
from traceback import clear_frames
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tts_tacotron2 import TtsModel
from fastapi.responses import FileResponse

app = FastAPI()

class Text(BaseModel):
    input : str


class ResponseAudi(BaseModel):
    output : list

@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    model_data_dir = Path("model_data")
    data = {}
    for f in model_data_dir.iterdir():
        # print(f)
        with open(f, "rb") as file:
            data[str(f)] = pickle.load(file)
    
    global model
    model = TtsModel(*data.values())


@app.get("/")
def home():
    return "Congratulations! HELLOYour API is working as expected. Now head over to http://localhost:80/docs"


@app.post("/predict", response_class=FileResponse)
def predict(text: Text):

    pred = model.predict(text.input)
    # pred = (pred.cpu().detach().numpy()[0]).tolist()
    # return {"Prediction": pred}
    return pred


# @app.get("/", tags=["Root"])
# async def read_root() -> dict:
#     return {
#         "message": "Welcome to my notes application, use the /docs route to proceeasdasdd"
    # }
