# Fine-Tuned SpeechT5 Text-to-Speech (TTS) Model

This repository provides an implementation of a fine-tuned version of the SpeechT5 Text-to-Speech (TTS) model, fine-tuned for English technical terms and Hindi regional language. The model can convert text input into synthesized speech for both languages, with a focus on technical vocabulary and regional accents.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Running the TTS Model](#running-the-tts-model)
    - [Testing and Evaluation](#testing-and-evaluation)
5. [Dataset](#dataset)
6. [Training and Fine-Tuning](#training-and-fine-tuning)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Contributing](#contributing)
10. [License](#license)

---

## Introduction

This project uses **Microsoft’s SpeechT5** model, a transformer-based architecture for Text-to-Speech tasks. It has been fine-tuned for two specific tasks:
- **English Technical Terms**: Specialized for accurate pronunciation of domain-specific words from fields like technology and science.
- **Hindi Regional Language**: Targeted at generating natural-sounding speech in Hindi, though improvements are needed.

The model was fine-tuned using datasets sourced from Hugging Face, and the fine-tuning process involved adjusting hyperparameters and training configurations.

---

## Features

- **Text-to-Speech Conversion**: Converts text input into speech for both English and Hindi.
- **Technical Term Pronunciation**: Improved handling of specialized terms in technology, engineering, and other domains.
- **Supports Fine-Tuning**: You can further fine-tune the model with your own dataset for different languages or domains.
- **Dataset Preparation and Preprocessing**: Tools provided for cleaning and preparing datasets before fine-tuning.

---

## Installation

Follow these steps to set up the environment and run the model:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Rohit-515/tts_finetuned_model.git
    cd tts_finetuned_model
    ```

2. **Create a Python Virtual Environment** (optional but recommended):
    ```bash
    python3 -m venv tts_env
    source tts_env/bin/activate
    ```

3. **Install Dependencies**:
    Install the required Python libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

    The key dependencies include:
    - `transformers`
    - `datasets`
    - `torchaudio`
    - `numpy`
    - `torch`

4. **Download Pre-Trained Model**:
    You can load the pre-trained model from Hugging Face using the following command inside your script:
    ```python
    from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
    model = SpeechT5ForTextToSpeech.from_pretrained('your-model-checkpoint')
    processor = SpeechT5Processor.from_pretrained('your-model-checkpoint')
    ```

---

## Usage

### Running the TTS Model

1. **Interactive TTS using Gradio**:
    The model comes with a simple Gradio interface for testing TTS you can create your own interface using the `app_reference_code.py` reference. To run the interface:
    
    ```bash
    python run_gradio_app.py
    ```

    You can input text and hear the generated speech.

2. **Using in Your Python Script**:
    You can directly use the model in a Python script to convert text to speech:

    ```python
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

    # Load fine-tuned model and processor
    model = SpeechT5ForTextToSpeech.from_pretrained('your-model-checkpoint')
    processor = SpeechT5Processor.from_pretrained('your-model-checkpoint')

    # Example text input
    text_input = "Enter your text here."

    # Preprocess text and convert to speech
    inputs = processor(text_input, return_tensors="pt")
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"])

    # Save the output as a .wav file
    torchaudio.save('output.wav', speech, 16000)
    ```

### Testing and Evaluation

You can test the model with your own text inputs or use the datasets provided to evaluate the model’s accuracy and naturalness. Evaluation metrics such as **Word Error Rate (WER)** and **Mean Opinion Score (MOS)** are recommended for assessing performance.

---

## Dataset

### Datasets Used for Fine-Tuning

1. **English Technical Terms**: Sourced from Hugging Face and focused on technical fields like technology and science. [Dataset Link](https://huggingface.co/datasets/Yassmen/TTS_English_Technical_data).
2. **Hindi Regional Language**: General words used in Hindi, though results were not fully satisfactory. [Dataset Link](https://huggingface.co/datasets/1rsh/tts-rj-hi-karya).

Feel free to substitute these datasets with your own to fine-tune the model on other domains or languages.

---

## Training and Fine-Tuning

To fine-tune the model with a custom dataset, follow these steps:

1. **Prepare Your Dataset**: Ensure your dataset is formatted correctly, with matching text and audio files. The dataset should be cleaned and preprocessed, including:
    - Text normalization (lowercasing, removing special characters).
    - Audio normalization (16kHz sampling rate).

2. **Fine-Tuning Process**:
    Modify the `fine_tune.py` script with your dataset paths and hyperparameter configurations.

    ```bash
    python fine_tune.py --dataset your-dataset-path --epochs 5 --batch_size 16 --learning_rate 3e-5
    ```

    For detailed instructions, refer to the fine-tuning documentation on [Hugging Face](https://huggingface.co/microsoft/speecht5_tts).

---

## Results

### Key Metrics:
- **Word Error Rate (WER)**:
  - English Technical Terms: 4.167%
  - Hindi Regional Language: 100%

- **Mean Opinion Score (MOS)**:
  - English Technical Terms: 3.66/5
  - Hindi Regional Language: 0/5

While the English model showed good accuracy for technical terms, the Hindi model needs further improvements, particularly with pronunciation and audio quality.

---

## Future Work

- **Improve Hindi TTS**: Refine the dataset and improve preprocessing techniques for better Hindi language performance.
- **Add More Languages**: Extend the model to other regional languages or domain-specific applications.
- **Optimize Model**: Experiment with training hyperparameters to enhance model performance.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests if you have improvements or new features to add.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

