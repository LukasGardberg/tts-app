import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from set_up_model import TtsModel
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os

app = FastAPI(title="Text-to-speech API")
current_path = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=f"{current_path}/templates")

class TextRequest(BaseModel):
    input : str


class AudioResponse(BaseModel):
    # What type is the response?
    # might need to be changed
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
def predict(text: TextRequest):

    # Model saves output wav in current directory and returns filename
    # Better way would probably be to not save it to memory,
    # but to stream it directly to the user (how?)
    # Just gonna try to load it from memory now
    file_name = clf.predict(text.input)

    # Get full filepath to send to FileResponse
    file_path = os.path.join(current_path, file_name)

    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    else:
        return {"error": "file not found"}


# Generates template for form
@app.get("/predict_form")
def predict(request: Request):
    return templates.TemplateResponse("form.html", context={"request": request})


@app.post("/predict_form", response_class=HTMLResponse)
def form_predict(text: TextRequest, request: Request):

    file_name = clf.predict(text.input)
    file_path = os.path.join(current_path, file_name)

    if os.path.exists(file_path):
        return templates.TemplateResponse("form.html", context={"request": request, "result": file_path})
    else:
        return {"error": "file not found"}


# Simple GET to generate audio without input
@app.get("/generate", response_class=FileResponse)
def generate():
    lines = ["Hello world", "Welcome to my API"]

    # get a random line from lines
    line = np.random.choice(lines)
    file_name = clf.predict(line)

    file_path = os.path.join(current_path, file_name)

    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    else:
        return {"error": "file not found"}
