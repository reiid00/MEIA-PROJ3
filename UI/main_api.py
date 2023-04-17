from flask import Flask, jsonify, request
import requests
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route('/resolveTicket', methods=['POST'])
def resolveTicket():

    # get the ticket text from the request
    ticket_text = request.json['ticket_text']

    to_lang = 'en'

    ticket_text_translated,detected_language= translate_text(ticket_text,to_lang)


    with ThreadPoolExecutor(max_workers=2) as executor:
        predict_categories_result= executor.submit(predict_categories, ticket_text_translated)
        predict_emotion_result = executor.submit(predict_emotion, ticket_text_translated)
        
        # prediction_result = predict_categories_result.result()
        emotions = predict_emotion_result.result()
        print("emotions",emotions)

        # prediction results
        typification = predict_categories_result.result()
        print("typification",typification)
        product= typification['product']
        sub_product= typification['sub_product']
        issue=typification['issue']
        sub_issue=typification['sub_issue']
   
        # print("",translated,"emotions",emotions)
        # product= "P"
        # sub_product= "SP"
        # issue="I"
        # sub_issue="SU"

    ticket_answer= generateTicketAnswer(ticket_text_translated, emotions, product, sub_product, issue, sub_issue)
    print("Main antes traducao final",ticket_answer)
    ticket_answer_translated= translate_text(ticket_answer,detected_language)
    # return the translated ticket text and detected language
    return jsonify({
        "ticket_text": ticket_text,
        "ticket_text_translated":ticket_text_translated,
        'ticket_answer_translated': ticket_answer_translated,
        'detected_language': detected_language,
        "ticket_answer":ticket_answer,
        "emotions":emotions,
        "product":product,
        "sub_product":sub_product,
        "issue":issue,
        "sub_issue":sub_issue,
        })

def translate_text(ticket_text,to_lang):
    # call the Translate API
    translate_url = 'http://localhost:8091/translate_text'
    params ={
        'ticket_text': ticket_text,
        'to_lang': to_lang
    }

    response = requests.post(translate_url, json=params)
 
    data = response.json()
    print(data.get('ticket_text_translated'),data.get('detected_language'))

    # return the translated ticket text and detected language
    return data.get('ticket_text_translated'),data.get('detected_language')

 
def generateTicketAnswer(ticket_text_translated, emotions, product, sub_product, issue, sub_issue):
    # call the Translate API
    generateTicketAnswer_url = 'http://localhost:8093/text_generation'
    params ={
        'ticket_text_translated': ticket_text_translated,
        'emotions':emotions,
        'product': product,
        'sub_product':sub_product,
        'issue':issue,
        'sub_issue':sub_issue
    }

    response = requests.post(generateTicketAnswer_url, json=params)
 
    data = response.json()
    print(data.get('ticket_answer'))
    
    # return the translated ticket text and detected language
    return data.get('ticket_answer')


def predict_categories(ticket_text_translated):
    # call the Translate API
    predict_categories_url = 'http://localhost:8092/predict_categories'
    params ={
        'ticket_text_translated': ticket_text_translated,
    }

    response = requests.post(predict_categories_url, json=params)
 
    data = response.json()
    
    # return the translated ticket text and detected language
    return data


def predict_emotion(ticket_text_translated):
    # call the Translate API
    predict_emotion_url = 'http://localhost:8090/predict_emotion'
    params ={
        'ticket_text_translated': ticket_text_translated,
    }

    response = requests.post(predict_emotion_url, json=params)
 
    data = response.json()

    # return the translated ticket text and detected language
    return data['emotions']
 
    
if __name__ == '__main__':
    app.run(host="localhost", port=8094, debug=True)