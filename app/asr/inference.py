"""
This script demonstrates the use of whisper model for processing audio input
and generating features using the whisper processor from hugging face transformer library 
and a dummy dataset from datasets
"""


from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# load model and processor
processor : WhisperProcessor = WhisperProcessor.from_pretrained("openai/whisper-small")
model : WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

# load dummy dataset and audio files 
dummy_dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample_audio = dummy_dataset[0]["audio"]

# process the audio input into features
input_features = processor(
    sample_audio["array"], 
    sampling_rate= sample_audio["sampling_rate"], 
    return_tensors="pt"
).input_features

# generate token ids
predicted_ids  = model.generate(input_features)

# decode predicted tokens to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# transcripted text
print(f"transcription: {transcription}")