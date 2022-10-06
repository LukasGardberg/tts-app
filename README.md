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

### Running with docker
Build the docker image with:

```
docker build -t tts-app .
```

Run the docker with:

```
docker run --rm -p 80:80 tts-app