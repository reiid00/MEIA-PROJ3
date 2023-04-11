from flask import Flask, request, jsonify
import torch
from DistilbertMultiLabel import DistilBertForMultilabelSequenceClassification
from transformers import AutoTokenizer
import numpy as np

model = "Reiid/go-emotional-analysis-distilbert"

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained(model)
model = DistilBertForMultilabelSequenceClassification.from_pretrained(model).to('cuda')
threshold = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/predict_emotion', methods=['POST'])
def predict():
    text = request.json['text']
    binary_preds = predict_label(text)
    return jsonify({'binary_preds': binary_preds.tolist()})

def predict_label(text, threshold=threshold, tokenizer=tokenizer, model=model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs= model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()[0]
    probs = softmax(logits)
    probs_scaled = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
    binary_preds = np.where(probs_scaled >= threshold, 1, 0)
    return binary_preds

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

if __name__ == '__main__':
    app.run(host="localhost", port=8090, debug=True)