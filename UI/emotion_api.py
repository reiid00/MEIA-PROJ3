from flask import Flask, request, jsonify
import torch
from DistilbertMultiLabel import DistilBertForMultilabelSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import json

model = "Reiid/go-emotional-analysis-distilbert"

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model)
model = DistilBertForMultilabelSequenceClassification.from_pretrained(model).to(device)
threshold = 0.5

with open('UI\data\i2label.json', 'r') as f:
    id2label = json.load(f)

@app.route('/predict_emotion', methods=['POST'])
def predict():
    text = request.json['text']
    binary_preds = predict_label(text)
    return jsonify({'emotions': binary_preds})

def predict_label(text, threshold=threshold, tokenizer=tokenizer, model=model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs= model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()[0]
    probs = softmax(logits)
    probs_scaled = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
    binary_preds = np.where(probs_scaled >= threshold)[0]
    return binary_preds_to_emotions(binary_preds)

def binary_preds_to_emotions(binary_preds):
    pred_emotions = [id2label[str(pred_id)]  for pred_id in binary_preds]
    return pred_emotions


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

if __name__ == '__main__':
    app.run(host="localhost", port=8090, debug=True)