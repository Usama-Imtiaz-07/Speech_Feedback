"""
Data Preparation for finetuning whisper-small used for Speech recognition
"""


from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio

# load dataset
common_voice_dataset = DatasetDict()
common_voice_dataset["train"] = load_dataset("mozilla-foundation/common-voice_11_0", "hi", split="train+validation")
common_voice_dataset["test"] = load_dataset("mozilla-foundation/common-voice_11_0", "hi", split="test")

# removing unnecessary columns (metadata info)
common_voice_dataset = common_voice_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

# WFE 1st pads the audio sample and then, 2nd converts the signals to log-mel spectogram
feature_extractor : WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# WT maps the token id sequence to text token eg. [1189, 8332, 3321] to > this is cat
tokenizer: WhisperTokenizer = WhisperTokenizers.from_pretrained("openai/whisper-small", language="hindi", task="transcribe")

# combining the WFE and WT in one   ? why
processor : WhisperProcessor = WhisperProcessor.from_pretrained("openai/whisper-small", language="hindi", task="transcribe")

# resampling to 16khz for whisper model
# optional ??
common_voice_dataset = common_voice_dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16 khz 
    audio = batch["audio"]

    # compute log-mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate = audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch 


common_voice_dataset = common_voice_dataset.map(
    prepare_dataset, 
    remove_columns=common_voice_dataset.column_names["train"], 
    num_proc=4
)

