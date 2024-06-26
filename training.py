"""
Adapted from: https://huggingface.co/blog/fine-tune-whisper
"""

from datasets import load_dataset, DatasetDict, Audio
from feature_extractor import CustomFeatureExtractor
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from huggingface_hub import login
login()
#%% Dataset loading

librispeech = DatasetDict()

librispeech["train"] = load_dataset("librispeech_asr", "clean", split="train.100+train.360")
librispeech["test"] = load_dataset("librispeech_asr", "clean", split="test")

librispeech = librispeech.remove_columns(['file', 'speaker_id', 'chapter_id', 'id'])


#%% Prepare Feature Extractor, Tokenizer and Data
feature_extractor = CustomFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor = WhisperProcessor(feature_extractor, tokenizer)

# %% Data preprocessing

librispeech = librispeech.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"].lower()).input_ids
    return batch

librispeech = librispeech.map(prepare_dataset, remove_columns=librispeech.column_names["train"], num_proc=6)

#%%
# ## Training and Evaluation

#%% Load a Pre-Trained Checkpoint
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# %% Define a Data Collator

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# %%
# ### Evaluation Metrics: We'll use the word error rate (WER) metric, the 'de-facto' metric for assessing
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

#%% Define the Training Configuration

training_args = Seq2SeqTrainingArguments(
    output_dir="./speech",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=1,
    gradient_checkpointing=False,
    dataloader_num_workers=6,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=2000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=librispeech["train"],
    eval_dataset=librispeech["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# %% #Training

# We'll save the processor object once before starting training. Since the processor is not trainable, it won't change over the course of training:
processor.save_pretrained(training_args.output_dir)

trainer.train()

# %% upload results
kwargs = {
    "dataset_tags": "librispeech_asr",
    "dataset": "librispeech_asr",  # a 'pretty' name for the training dataset
    "dataset_args": "config: clean, split: train",
    "language": "en",
    "model_name": "SpeechGPT",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}

trainer.push_to_hub(**kwargs)