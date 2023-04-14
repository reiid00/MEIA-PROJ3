from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
from flask import Flask, request, jsonify
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)

tokenizer_pt_en = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5")

model_pt_en = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5").to(device)

tokenizer_en_pt = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")

model_en_pt = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt").to(device)

def load_translation_model(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    return tokenizer, model

def detect_language(text):
    try:
        detected_lang = detect(text)
        return detected_lang
    except:
        raise ValueError("Language could not be detected")

def translate(text, src_lang, tgt_lang, model = None, tokenizer = None, max_new_tokens = 512):
    # Verify if is the same language
    if src_lang == tgt_lang:
        return text
    if model is None:
        # Load the translation model
        tokenizer, model = load_translation_model(src_lang, tgt_lang)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Perform the translation
    outputs = model.generate(**inputs,max_new_tokens=max_new_tokens)

    # Decode the translated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text

def handle_translation(text, target_language):
    try:
        # Detect the source language
        src_lang = detect_language(text)
        model = None
        tokenizer = None
        if src_lang == "pt":
            model = model_pt_en
            tokenizer = tokenizer_pt_en
        elif target_language == "pt":
            model = model_en_pt
            tokenizer = tokenizer_en_pt
        translated_text = translate(text, src_lang, target_language, model, tokenizer)
        return translated_text,src_lang
    except ValueError as e:
        return str(e)

@app.route('/translate_text', methods=['POST'])
def translate_text():
    print("translate_text")
    text = request.json['ticket_text']
    lang = request.json['to_lang']
    print("text",text,"lang",lang)
    translated_text,src_lang = handle_translation(text, lang)
    print(translated_text,src_lang)
    return jsonify({'ticket_text_translated': translated_text, 'detected_language': src_lang})
    


if __name__ == '__main__':
    app.run(host="localhost", port=8091, debug=True)