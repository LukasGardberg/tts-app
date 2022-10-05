import torch
import torchaudio
import pickle
import os

import IPython

class TtsModel():
    def __init__(self):
        self.bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = self.bundle.get_text_processor()
        self.tacotron2 = self.bundle.get_tacotron2().to(self.device)
        self.vocoder = self.bundle.get_vocoder().to(self.device)

    
    def predict(self, text):
        with torch.inference_mode():
            processed, lengths = self.processor(text)
            processed = processed.to(self.device)
            lengths = lengths.to(self.device)
            spec, spec_lengths, _ = self.tacotron2.infer(processed, lengths)
            waveforms, lengths = self.vocoder(spec, spec_lengths)
        filename = text.replace(' ', '_')
        output_filepath = f"{filename}.wav"
        torchaudio.save(output_filepath, waveforms[0:1].cpu(), sample_rate=self.vocoder.sample_rate)
        #IPython.display.display(IPython.display.Audio("output_wavernn.wav"))
        return output_filepath






model = TtsModel()
model_name = 'tacotron'
if not os.path.isfile(f"{model_name}.pkl"):
    with open(f'{model_name}.pkl', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    