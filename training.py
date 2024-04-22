# %% [markdown]
# # Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers

# %% [markdown]
# In this Colab, we present a step-by-step guide on how to fine-tune Whisper
# for any multilingual ASR dataset using Hugging Face 🤗 Transformers. This is a
# more "hands-on" version of the accompanying [blog post](https://huggingface.co/blog/fine-tune-whisper).
# For a more in-depth explanation of Whisper, the Common Voice dataset and the theory behind fine-tuning, the reader is advised to refer to the blog post.

# %% [markdown]
# ## Introduction

# %% [markdown]
# Whisper is a pre-trained model for automatic speech recognition (ASR)
# published in [September 2022](https://openai.com/blog/whisper/) by the authors
# Alec Radford et al. from OpenAI. Unlike many of its predecessors, such as
# [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477), which are pre-trained
# on un-labelled audio data, Whisper is pre-trained on a vast quantity of
# **labelled** audio-transcription data, 680,000 hours to be precise.
# This is an order of magnitude more data than the un-labelled audio data used
# to train Wav2Vec 2.0 (60,000 hours). What is more, 117,000 hours of this
# pre-training data is multilingual ASR data. This results in checkpoints
# that can be applied to over 96 languages, many of which are considered
# _low-resource_.
# 
# When scaled to 680,000 hours of labelled pre-training data, Whisper models
# demonstrate a strong ability to generalise to many datasets and domains.
# The pre-trained checkpoints achieve competitive results to state-of-the-art
# ASR systems, with near 3% word error rate (WER) on the test-clean subset of
# LibriSpeech ASR and a new state-of-the-art on TED-LIUM with 4.7% WER (_c.f._
# Table 8 of the [Whisper paper](https://cdn.openai.com/papers/whisper.pdf)).
# The extensive multilingual ASR knowledge acquired by Whisper during pre-training
# can be leveraged for other low-resource languages; through fine-tuning, the
# pre-trained checkpoints can be adapted for specific datasets and languages
# to further improve upon these results. We'll show just how Whisper can be fine-tuned
# for low-resource languages in this Colab.

# %% [markdown]
# <figure>
# <img src="https://raw.githubusercontent.com/sanchit-gandhi/notebooks/main/whisper_architecture.svg" alt="Trulli" style="width:100%">
# <figcaption align = "center"><b>Figure 1:</b> Whisper model. The architecture
# follows the standard Transformer-based encoder-decoder model. A
# log-Mel spectrogram is input to the encoder. The last encoder
# hidden states are input to the decoder via cross-attention mechanisms. The
# decoder autoregressively predicts text tokens, jointly conditional on the
# encoder hidden states and previously predicted tokens. Figure source:
# <a href="https://openai.com/blog/whisper/">OpenAI Whisper Blog</a>.</figcaption>
# </figure>

# %% [markdown]
# The Whisper checkpoints come in five configurations of varying model sizes.
# The smallest four are trained on either English-only or multilingual data.
# The largest checkpoints are multilingual only. All 11 of the pre-trained checkpoints
# are available on the [Hugging Face Hub](https://huggingface.co/models?search=openai/whisper). The
# checkpoints are summarised in the following table with links to the models on the Hub:
# 
# | Size     | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                        |
# |----------|--------|-------|-------|------------|------------------------------------------------------|-----------------------------------------------------|
# | tiny     | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny.)    |
# | base     | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)     |
# | small    | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)    |
# | medium   | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium)   |
# | large    | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)    |
# | large-v2 | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v2) |
# | large-v3 | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v3) |
# 
# 
# For demonstration purposes, we'll fine-tune the multilingual version of the
# [`"small"`](https://huggingface.co/openai/whisper-small) checkpoint with 244M params (~= 1GB).
# As for our data, we'll train and evaluate our system on a low-resource language
# taken from the [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
# dataset. We'll show that with as little as 8 hours of fine-tuning data, we can achieve
# strong performance in this language.

# %% [markdown]
# ------------------------------------------------------------------------
# 
# \\({}^1\\) The name Whisper follows from the acronym “WSPSR”, which stands for “Web-scale Supervised Pre-training for Speech Recognition”.

# %% [markdown]
# ## Prepare Environment

# %% [markdown]
# First of all, let's try to secure a decent GPU for our Colab! Unfortunately, it's becoming much harder to get access to a good GPU with the free version of Google Colab. However, with Google Colab Pro one should have no issues in being allocated a V100 or P100 GPU.
# 
# To get a GPU, click _Runtime_ -> _Change runtime type_, then change _Hardware accelerator_ from _CPU_ to one of the available GPUs, e.g. _T4_ (or better if you have one available). Next, click `Connect T4` in the top right-hand corner of your screen (or `Connect {V100, A100}` if you selected a different GPU).

# %% [markdown]
# We can verify that we've been assigned a GPU and view its specifications:

# %% [markdown]
# We'll employ several popular Python packages to fine-tune the Whisper model.
# We'll use `datasets[audio]` to download and prepare our training data, alongside
# `transformers` and `accelerate` to load and train our Whisper model.
# We'll also require the `soundfile` package to pre-process audio files,
# `evaluate` and `jiwer` to assess the performance of our model, and
# `tensorboard` to log our metrics. Finally, we'll use `gradio` to build a
# flashy demo of our fine-tuned model.

# %% [markdown]
# We strongly advise you to upload model checkpoints directly the [Hugging Face Hub](https://huggingface.co/)
# whilst training. The Hub provides:
# - Integrated version control: you can be sure that no model checkpoint is lost during training.
# - Tensorboard logs: track important metrics over the course of training.
# - Model cards: document what a model does and its intended use cases.
# - Community: an easy way to share and collaborate with the community!
# 
# Linking the notebook to the Hub is straightforward - it simply requires entering your
# Hub authentication token when prompted. Find your Hub authentication token [here](https://huggingface.co/settings/tokens):

# %%
from huggingface_hub import login

login()

# %% [markdown]
# ## Load Dataset

# %% [markdown]
# Using 🤗 Datasets, downloading and preparing data is extremely simple.
# We can download and prepare the Common Voice splits in just one line of code.
# 
# First, ensure you have accepted the terms of use on the Hugging Face Hub: [mozilla-foundation/common_voice_11_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0). Once you have accepted the terms, you will have full access to the dataset and be able to download the data locally.
# 
# Since Hindi is very low-resource, we'll combine the `train` and `validation`
# splits to give approximately 8 hours of training data. We'll use the 4 hours
# of `test` data as our held-out test set:

# %%
from datasets import load_dataset, DatasetDict, DownloadConfig

librispeech = DatasetDict()

librispeech["train"] = load_dataset("librispeech_asr", "clean", split="train.100+train.360")
librispeech["test"] = load_dataset("librispeech_asr", "clean", split="test")

print(librispeech)

# %% [markdown]
# Most ASR datasets only provide input audio samples (`audio`) and the
# corresponding transcribed text (`sentence`). Common Voice contains additional
# metadata information, such as `accent` and `locale`, which we can disregard for ASR.
# Keeping the notebook as general as possible, we only consider the input audio and
# transcribed text for fine-tuning, discarding the additional metadata information:

# %%
librispeech = librispeech.remove_columns(['file', 'speaker_id', 'chapter_id', 'id'])

print(librispeech)

# %% [markdown]
# ## Prepare Feature Extractor, Tokenizer and Data

# %% [markdown]
# The ASR pipeline can be de-composed into three stages:
# 
# 1. A feature extractor which pre-processes the raw audio-inputs
# 2. The model which performs the sequence-to-sequence mapping
# 3. A tokenizer which post-processes the model outputs to text format
# 
# In 🤗 Transformers, the Whisper model has an associated feature extractor and tokenizer,
# called [WhisperFeatureExtractor](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperFeatureExtractor)
# and [WhisperTokenizer](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperTokenizer)
# respectively.
# 
# We'll go through details for setting-up the feature extractor and tokenizer one-by-one!

# %% [markdown]
# ### Load WhisperFeatureExtractor

# %% [markdown]
# The Whisper feature extractor performs two operations:
# 1. Pads / truncates the audio inputs to 30s: any audio inputs shorter than 30s are padded to 30s with silence (zeros), and those longer that 30s are truncated to 30s
# 2. Converts the audio inputs to _log-Mel spectrogram_ input features, a visual representation of the audio and the form of the input expected by the Whisper model

# %% [markdown]
# <figure>
# <img src="https://raw.githubusercontent.com/sanchit-gandhi/notebooks/main/spectrogram.jpg" alt="Trulli" style="width:100%">
# <figcaption align = "center"><b>Figure 2:</b> Conversion of sampled audio array to log-Mel spectrogram.
# Left: sampled 1-dimensional audio signal. Right: corresponding log-Mel spectrogram. Figure source:
# <a href="https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html">Google SpecAugment Blog</a>.
# </figcaption>

# %% [markdown]
# We'll load the feature extractor from the pre-trained checkpoint with the default values:

# %%
# from transformers import WhisperFeatureExtractor

# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# %%
from feature_extractor import CustomFeatureExtractor
feature_extractor = CustomFeatureExtractor.from_pretrained("openai/whisper-small")

# %% [markdown]
# ### Load WhisperTokenizer

# %% [markdown]
# The Whisper model outputs a sequence of _token ids_. The tokenizer maps each of these token ids to their corresponding text string. For Hindi, we can load the pre-trained tokenizer and use it for fine-tuning without any further modifications. We simply have to
# specify the target language and the task. These arguments inform the
# tokenizer to prefix the language and task tokens to the start of encoded
# label sequences:

# %%
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
# %% [markdown]
# ### Combine To Create A WhisperProcessor

# %% [markdown]
# To simplify using the feature extractor and tokenizer, we can _wrap_
# both into a single `WhisperProcessor` class. This processor object
# inherits from the `WhisperFeatureExtractor` and `WhisperProcessor`,
# and can be used on the audio inputs and model predictions as required.
# In doing so, we only need to keep track of two objects during training:
# the `processor` and the `model`:

# %%
from transformers import WhisperProcessor

processor = WhisperProcessor(feature_extractor, tokenizer)

# %% [markdown]
# ### Prepare Data

# %% [markdown]
# Let's print the first example of the Common Voice dataset to see
# what form the data is in:

# %% [markdown]
# Since
# our input audio is sampled at 48kHz, we need to _downsample_ it to
# 16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model.
# 
# We'll set the audio inputs to the correct sampling rate using dataset's
# [`cast_column`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=cast_column#datasets.DatasetDict.cast_column)
# method. This operation does not change the audio in-place,
# but rather signals to `datasets` to resample audio samples _on the fly_ the
# first time that they are loaded:

# %%
from datasets import Audio

librispeech = librispeech.cast_column("audio", Audio(sampling_rate=16000))

# %% [markdown]
# Re-loading the first audio sample in the Common Voice dataset will resample
# it to the desired sampling rate:

# %%
print(librispeech["train"][0])

# %% [markdown]
# Now we can write a function to prepare our data ready for the model:
# 1. We load and resample the audio data by calling `batch["audio"]`. As explained above, 🤗 Datasets performs any necessary resampling operations on the fly.
# 2. We use the feature extractor to compute the log-Mel spectrogram input features from our 1-dimensional audio array.
# 3. We encode the transcriptions to label ids through the use of the tokenizer.

# %%
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"].lower()).input_ids
    return batch

# %% [markdown]
# We can apply the data preparation function to all of our training examples using dataset's `.map` method. The argument `num_proc` specifies how many CPU cores to use. Setting `num_proc` > 1 will enable multiprocessing. If the `.map` method hangs with multiprocessing, set `num_proc=1` and process the dataset sequentially.

# %%
import multiprocessing as mp
librispeech = librispeech.map(prepare_dataset, remove_columns=librispeech.column_names["train"], num_proc=6)

# %% [markdown]
# ## Training and Evaluation

# %% [markdown]
# Now that we've prepared our data, we're ready to dive into the training pipeline.
# The [🤗 Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer)
# will do much of the heavy lifting for us. All we have to do is:
# 
# - Load a pre-trained checkpoint: we need to load a pre-trained checkpoint and configure it correctly for training.
# 
# - Define a data collator: the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.
# 
# - Evaluation metrics: during evaluation, we want to evaluate the model using the [word error rate (WER)](https://huggingface.co/metrics/wer) metric. We need to define a `compute_metrics` function that handles this computation.
# 
# - Define the training configuration: this will be used by the 🤗 Trainer to define the training schedule.
# 
# Once we've fine-tuned the model, we will evaluate it on the test data to verify that we have correctly trained it
# to transcribe speech in Hindi.

# %% [markdown]
# ### Load a Pre-Trained Checkpoint

# %% [markdown]
# We'll start our fine-tuning run from the pre-trained Whisper `small` checkpoint,
# the weights for which we need to load from the Hugging Face Hub. Again, this
# is trivial through use of 🤗 Transformers!

# %%
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# %% [markdown]
# We can disable the automatic language detection task performed during inference, and force the model to generate in Hindi. To do so, we set the [langauge](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.language)
# and [task](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.task)
# arguments to the generation config. We'll also set any [`forced_decoder_ids`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids)
# to None, since this was the legacy way of setting the language and
# task arguments:

# %%
model.generation_config.language = "english"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

# %% [markdown]
# ### Define a Data Collator

# %% [markdown]
# The data collator for a sequence-to-sequence speech model is unique in the sense that it
# treats the `input_features` and `labels` independently: the  `input_features` must be
# handled by the feature extractor and the `labels` by the tokenizer.
# 
# The `input_features` are already padded to 30s and converted to a log-Mel spectrogram
# of fixed dimension by action of the feature extractor, so all we have to do is convert the `input_features`
# to batched PyTorch tensors. We do this using the feature extractor's `.pad` method with `return_tensors=pt`.
# 
# The `labels` on the other hand are un-padded. We first pad the sequences
# to the maximum length in the batch using the tokenizer's `.pad` method. The padding tokens
# are then replaced by `-100` so that these tokens are **not** taken into account when
# computing the loss. We then cut the BOS token from the start of the label sequence as we
# append it later during training.
# 
# We can leverage the `WhisperProcessor` we defined earlier to perform both the
# feature extractor and the tokenizer operations:

# %%
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from dataclasses import dataclass
from typing import Any, Dict, List, Union

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

# %% [markdown]
# Let's initialise the data collator we've just defined:

# %%
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# %% [markdown]
# ### Evaluation Metrics

# %% [markdown]
# We'll use the word error rate (WER) metric, the 'de-facto' metric for assessing
# ASR systems. For more information, refer to the WER [docs](https://huggingface.co/metrics/wer). We'll load the WER metric from 🤗 Evaluate:

# %%
import evaluate

metric = evaluate.load("wer")

# %% [markdown]
# We then simply have to define a function that takes our model
# predictions and returns the WER metric. This function, called
# `compute_metrics`, first replaces `-100` with the `pad_token_id`
# in the `label_ids` (undoing the step we applied in the
# data collator to ignore padded tokens correctly in the loss).
# It then decodes the predicted and label ids to strings. Finally,
# it computes the WER between the predictions and reference labels:

# %%
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

# %% [markdown]
# ### Define the Training Configuration

# %% [markdown]
# In the final step, we define all the parameters related to training. For more detail on the training arguments, refer to the Seq2SeqTrainingArguments [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

# %%
from transformers import Seq2SeqTrainingArguments

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

# %% [markdown]
# **Note**: if one does not want to upload the model checkpoints to the Hub,
# set `push_to_hub=False`.

# %% [markdown]
# We can forward the training arguments to the 🤗 Trainer along with our model,
# dataset, data collator and `compute_metrics` function:

# %%
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=librispeech["train"],
    eval_dataset=librispeech["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# %% [markdown]
# We'll save the processor object once before starting training. Since the processor is not trainable, it won't change over the course of training:

# %%
processor.save_pretrained(training_args.output_dir)

# %% [markdown]
# ### Training

# %% [markdown]
# Training will take approximately 5-10 hours depending on your GPU or the one
# allocated to this Google Colab. If using this Google Colab directly to
# fine-tune a Whisper model, you should make sure that training isn't
# interrupted due to inactivity. A simple workaround to prevent this is
# to paste the following code into the console of this tab (_right mouse click_
# -> _inspect_ -> _Console tab_ -> _insert code_).

# %% [markdown]
# ```javascript
# function ConnectButton(){
#     console.log("Connect pushed");
#     document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click()
# }
# setInterval(ConnectButton, 60000);
# ```

# %% [markdown]
# The peak GPU memory for the given training configuration is approximately 15.8GB.
# Depending on the GPU allocated to the Google Colab, it is possible that you will encounter a CUDA `"out-of-memory"` error when you launch training.
# In this case, you can reduce the `per_device_train_batch_size` incrementally by factors of 2
# and employ [`gradient_accumulation_steps`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments.gradient_accumulation_steps)
# to compensate.
# 
# To launch training, simply execute:

# %%
trainer.train()

# %% [markdown]
# Our best WER is 32.0% - not bad for 8h of training data! We can make our model more accessible on the Hub with appropriate tags and README information.
# You can change these values to match your dataset, language and model
# name accordingly:

# %%
kwargs = {
    "dataset_tags": "librispeech_asr",
    "dataset": "librispeech_asr",  # a 'pretty' name for the training dataset
    "dataset_args": "config: clean, split: train",
    "language": "en",
    "model_name": "SpeechGPT",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}

# %% [markdown]
# The training results can now be uploaded to the Hub. To do so, execute the `push_to_hub` command and save the preprocessor object we created:

# %%
trainer.push_to_hub(**kwargs)