import torch
import torchaudio

# Attempt at creating a single function that
# takes in text and outputs a wav file

# The goal is to be able to do something like
# from models.tts_tacotron2 import generate_wav
# in an API 

device = "cuda" if torch.cuda.is_available() else "cpu"

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
text_processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2()
vocoder = bundle.get_vocoder()

def generate_wav(text: str):

    with torch.inference_mode():
        processed, lengths = text_processor(text)
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        waveforms, lengths = vocoder(spec, spec_lengths)

    torchaudio.save("output_wavernn.wav", waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)
    # What should be returned to the front end?
