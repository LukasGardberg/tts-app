import torch
import torchaudio

# Attempt at creating a single function that
# takes in text and outputs a wav file

# The goal is to be able to do something like
# from models.tts_tacotron2 import generate_wav
# in an API 


class TtsModel():
    def __init__(self, bundle, processor, model, vocoder):
        self.bundle = bundle
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = processor
        self.tacotron2 = model
        self.vocoder = vocoder
        

    
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
        return output_filepath