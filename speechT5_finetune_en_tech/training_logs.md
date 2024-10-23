---
library_name: transformers
license: mit
base_model: microsoft/speecht5_tts
tags:
- tts
- generated_from_trainer
datasets:
- microsoft/speecht5_tts
model-index:
- name: SpeechT5 Technical English
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# SpeechT5 Technical English

This model is a fine-tuned version of [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) on the TTS_English_Technical_data dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4566

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 4
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 100
- training_steps: 500
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 4.6769        | 0.3581 | 100  | 0.5191          |
| 4.3319        | 0.7162 | 200  | 0.4869          |
| 4.1006        | 1.0743 | 300  | 0.4700          |
| 4.0038        | 1.4324 | 400  | 0.4597          |
| 3.9472        | 1.7905 | 500  | 0.4566          |


### Framework versions

- Transformers 4.46.0.dev0
- Pytorch 2.4.1+cu121
- Datasets 3.0.1
- Tokenizers 0.20.1
