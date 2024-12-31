"""
Evaluating Whisper-small with librispeech/clean/test dataset
on the basis of Word Error Rate metric, before finetuning the LLM. 
"""


from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load


# pre-training evaluation

# loading dataset
librispeech_test_clean_dataset = load_dataset("librispeech_asr", "clean", split = "test")

processor : WhisperProcessor = WhisperProcessor.from_pretrained("openai/whisper-small")
model : WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to('cuda')

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(
    audio['array'], 
    sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    batch["reference"] = processor.tokenizer._normalize(batch['text'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch

result = librispeech_test_clean_dataset.map(map_to_pred)


# word error rate metric evaluation
wer = load("wer")
print(100* wer.compute(references=results["reference"], predictions=result["prediction"]))
