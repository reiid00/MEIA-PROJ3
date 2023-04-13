from flask import Flask, jsonify, request
import requests
import json

app = Flask(__name__)

@app.route('/resolveTicket', methods=['POST'])
def resolveTicket():
    print("resolveTicket")
    # get the ticket text from the request
    ticket_text = request.json['ticket_text']
    print("ticket_text",ticket_text)
    # call the Translate API
    translate_url = 'http://localhost:8091/translate_text'
    params ={
        'ticket_text': ticket_text,
        'to_lang': 'pt'
    }

    response = requests.post(translate_url, json=params)
    print("response2----------")
    print("response",response.json())
    data = response.json()
    print(data.get('ticket_text_translated'),data.get('detected_language'))
    # return the translated ticket text and detected language
    return jsonify({
        'ticket_text_translated': data.get('ticket_text_translated'),
        'detected_language': data.get('detected_language')
    })

    


if __name__ == '__main__':
    app.run(host="localhost", port=8094, debug=True)