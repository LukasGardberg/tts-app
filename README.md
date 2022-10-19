### Start a virtual environment and install necessary packages
```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Links to relevant information

https://pytorch.org/tutorials/intermediate/text_to_speech_with_torchaudio.html# tts-app

test


### Running the flask app

Navigate to the 'app' directory. Create the model tacotron by running
```
python3 set_up_model.py
```

Run the application by running

```
uvicorn main:app --reload
```

To try and generate a pre-defined audio clip, go to localhost/generate (e.g http://127.0.0.1:8000/generate).
This will prompt a file download.

Detta kanske är något? https://stackoverflow.com/questions/73234675/how-to-download-a-file-after-posting-data-using-fastapi