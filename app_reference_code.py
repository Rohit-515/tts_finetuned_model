import gradio as gr
import torch
import soundfile as sf
import os
import numpy as np
import re
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models_and_data():
    model_name = "microsoft/speecht5_tts"
    processor = SpeechT5Processor.from_pretrained(model_name)
    model = SpeechT5ForTextToSpeech.from_pretrained("your_model_name").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", spk_model_name),
    )
    
    # Load a sample from a dataset for default embedding
    dataset = load_dataset("your_example_dataset", split="train")
    example = dataset[304]
    
    return model, processor, vocoder, speaker_model, example

model, processor, vocoder, speaker_model, default_example = load_models_and_data()

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform).unsqueeze(0).to(device))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze()
    return speaker_embeddings

def prepare_default_embedding(example):
    audio = example["audio"]
    return create_speaker_embedding(audio["array"])

default_embedding = prepare_default_embedding(default_example)

# define your own replacements for unrecognized character by the base model, some example for replacement character is given down below

replacements = [
    ("â", "a"),  # Long a
    ("ç", "ch"),  # Ch as in "chair"
    ("ğ", "gh"),  # Silent g or slight elongation of the preceding vowel
   #........
]


# define a function/logic to convert numbers into words in input text you can use the inbuit libraries like num2words for english.
def number_to_words(number):
    #your code


def replace_numbers_with_words(text):
    def replace(match):
        number = int(match.group())
        return number_to_words(number)

    # Find the numbers and change with words.
    result = re.sub(r'\b\d+\b', replace, text)

    return result

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Replace numbers with words
    text = replace_numbers_with_words(text)
    
    # Apply character replacements
    for old, new in replacements:
        text = text.replace(old, new)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def text_to_speech(text, audio_file=None):
    # Normalize the input text
    normalized_text = normalize_text(text)
    
    # Prepare the input for the model
    inputs = processor(text=normalized_text, return_tensors="pt").to(device)
    
    # Use the default speaker embedding
    speaker_embeddings = default_embedding
    
    # Generate speech
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings.unsqueeze(0), vocoder=vocoder)
    
    speech_np = speech.cpu().numpy()

    return (16000, speech_np)
    
iface = gr.Interface(
    fn=text_to_speech,
    inputs=[
        gr.Textbox(label="Enter Turkish text to convert to speech")
    ],
    outputs=[
        gr.Audio(label="Generated Speech", type="numpy")
    ],
    title="Turkish SpeechT5 Text-to-Speech Demo",
    description="Enter Turkish text, and listen to the generated speech."
)

iface.launch(share=True)
