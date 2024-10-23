---
library_name: transformers
license: mit
base_model: microsoft/speecht5_tts
tags:
- generated_from_trainer
model-index:
- name: speecht5_finetuned_rohit_hindi
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# speecht5_finetuned_rohit_hindi

This model is a fine-tuned version of [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5786

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 4
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 100
- training_steps: 800
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.8562        | 0.3366 | 100  | 0.7391          |
| 0.7663        | 0.6731 | 200  | 0.6572          |
| 0.7168        | 1.0097 | 300  | 0.6354          |
| 0.6915        | 1.3462 | 400  | 0.6043          |
| 0.6651        | 1.6828 | 500  | 0.6000          |
| 0.6573        | 2.0194 | 600  | 0.5836          |
| 0.6347        | 2.3559 | 700  | 0.5818          |
| 0.6554        | 2.6925 | 800  | 0.5786          |


### Framework versions

- Transformers 4.44.2
- Pytorch 2.4.1+cu121
- Datasets 3.0.2
- Tokenizers 0.19.1
