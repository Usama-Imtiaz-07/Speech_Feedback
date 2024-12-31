from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# load model and tokenizer
model_name = "tabularisai/robust-sentiment-analysis"
tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
model : AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(model_name)

# predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(
        text.lower(), 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    with torch.no_grad():
        outputs=model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return sentiment_map[predicted_class]

# example usage
text = [
    "I absolutely loved this movie! The acting was superb and the plot was engaging.",
    "The service at this restaurant was terrible. I'll never go back.",
    "The product works as expected. Nothing special, but it gets the job done.",
    "I'm somewhat disappointed with my purchase. It's not as good as I hoped.",
    "This book changed my life! I couldn't put it down and learned so much."
]

for line in text:
    sentiment = predict_sentiment(line)
    print(f"Text: {line}")
    print(f"Sentiment: {sentiment}\n")