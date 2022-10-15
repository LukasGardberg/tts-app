import torch
import torchaudio
import pickle
import os
from pathlib import Path

import IPython

class Setup:
    def __init__(self):
        self.data = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def download(self):
        self.data["bundle"] =  torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        self.data["processor"] = self.data["bundle"].get_text_processor()
        self.data["tacotron2"] = self.data["bundle"].get_tacotron2().to(self.device)
        self.data["vocoder"] = self.data["bundle"].get_vocoder().to(self.device)


    def save_locally(self):
        model_data_dir = Path("model_data")
        model_data_dir.mkdir(exist_ok=True)
        for key, val in self.data.items():
            if not os.path.isfile(f"{model_data_dir}/{key}.pkl"):
                with open(f'{model_data_dir}/{key}.pkl', 'wb') as handle:
                    pickle.dump(val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


setup = Setup()
setup.download()
setup.save_locally()

    
