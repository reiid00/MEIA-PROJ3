from flask import Flask, request, jsonify
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import json
# Uncomment if not previously installed
#import nltk
#nltk.download('stopwords')
# import nltk
# nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
PLACEHOLDERS_RE = re.compile(r'\bx+\b')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = PLACEHOLDERS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def clean_text_data(data):
    data = data.apply(clean_text)
    data = data.str.replace('\d+', '', regex=True)
    return data

def preprocess_text_data(data, preproc_option=None, max_words=50000, max_words_complaint=2500, vector_size=100, ignore_comments=True):
    data = clean_text_data(data).values
    
    if preproc_option == 'stemming':
        stemmer = SnowballStemmer('english')
        data = [' '.join(stemmer.stem(word) for word in complaint.split()) for complaint in data]
    elif preproc_option == 'lemmatization':
        lemmatizer = WordNetLemmatizer()
        data = [' '.join(lemmatizer.lemmatize(word) for word in complaint.split()) for complaint in data]

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>", lower=True)
    tokenizer.fit_on_texts(data)

    if ignore_comments == False:
        print('Found %s unique tokens.' % len(tokenizer.word_index))

    X = tokenizer.texts_to_sequences(data)
    X = pad_sequences(X, maxlen=max_words_complaint)

    return np.array(X)

# Load model function
def loadModel(path):
    json_file = open(f'{path}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f'{path}.h5')
    return loaded_model

def loadID2Labels():
    with open('./UI/data/id2label_product.json', 'r') as f:
        id2label_product = json.load(f)
    with open('./UI/data/id2label_sub_product.json', 'r') as f:
        id2label_sub_product = json.load(f)
    with open('./UI/data/id2label_issue.json', 'r') as f:
        id2label_issue = json.load(f)
    with open('./UI/data/id2label_sub_issue.json', 'r') as f:
        id2label_sub_issue = json.load(f)
    return id2label_product, id2label_sub_product, id2label_issue, id2label_sub_issue

# Load ID2Labels to use for Labelling
id2label_product, id2label_sub_product, id2label_issue, id2label_sub_issue = loadID2Labels()

# Load Chosen Model
chosen_model_path = "./Typification/saved_models/model_cnn_multi_none"
chosen_model = loadModel(chosen_model_path)

app = Flask(__name__)

@app.route('/predict_categories', methods=['POST'])
def predict():
    new_complaint = request.json['ticket_text_translated']
    pred = chosen_model.predict(preprocess_text_data(pd.Series(new_complaint)))
    return jsonify({
        'product': id2label_product[str(np.argmax(pred[0]))], 
        'sub_product': id2label_sub_product[str(np.argmax(pred[1]))], 
        'issue': id2label_issue[str(np.argmax(pred[2]))], 
        'sub_issue': id2label_sub_issue[str(np.argmax(pred[3]))]
        })

if __name__ == '__main__':
    app.run(host="localhost", port=8092, debug=True)