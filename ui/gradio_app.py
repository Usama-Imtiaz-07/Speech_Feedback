import gradio as gr
from transformers import pipeline

# Load models
asr_model = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h")
sentiment_model = pipeline("sentiment-analysis")

def process_audio(audio):
    text = asr_model(audio)["text"]
    sentiment = sentiment_model(text)[0]
    return text, sentiment['label'], sentiment['score']

demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Record or Upload Audio"),
    outputs=[
        gr.Text(label="Transcription"),
        gr.Label(label="Sentiment"),
        gr.Number(label="Confidence"),
    ],
    title="ASR + Sentiment Analysis",
    description="Upload or record audio, and the app will transcribe and analyze its sentiment.",
)

demo.launch()
