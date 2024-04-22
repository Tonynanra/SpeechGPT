from transformers import pipeline
import gradio as gr
import os
import numpy as np
import soundfile as sf
import torch
from feature_extractor import CustomFeatureExtractor, WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("tluo23/speech")

pipe = pipeline(
    task="automatic-speech-recognition",
    model="tluo23/speech",
    feature_extractor=feature_extractor,
    device='cuda' if torch.cuda.is_available() else 'cpu', 
    )  # change to "your-username/the-name-you-picked"


def transcribe(audio, state=""):
    data, _ = sf.read(audio)
    rms = np.sqrt(np.mean(data**2))
    
    volume_threshold = 0.01
    
    if rms > volume_threshold:
        text = pipe(audio)["text"]
        state += text + ' '

    os.remove(audio)

    return state, state

iface = gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(sources="microphone", type="filepath", streaming=True), "state"],
    outputs=["text","state"],
    title="SpeechGPT",
    live=True,
    description="Automatic Speech Recognition using Finetuned Transformer Models. Speak into the microphone to get started!",
)

iface.launch(share=True)