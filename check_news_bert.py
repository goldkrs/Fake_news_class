import torch
from transformers.models.distilbert import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model
model = DistilBertForSequenceClassification.from_pretrained("bert_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("bert_model")
model.eval()

# User input
print(" Enter the news title:")
title = input("> ")
print(" Enter the news body/content:")
body = input("> ")

text = title + " " + body

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# Output
print("\n Prediction:")
print("REAL NEWS" if predicted_class == 0 else "FAKE NEWS")
