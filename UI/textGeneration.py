
from flask import Flask, request, jsonify
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline



import openai
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)

# Set up OpenAI API credentials and create a GPT-2 model instance
openai.api_key = "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"
model="gpt-3.5-turbo"

def generate(prompt,model):

    prompt_msg = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
    generated_text = prompt_msg.choices[0]['message']
    message_prompt = generated_text['content']

    return message_prompt


def generateText(text, emotions, context):
    try:

        #Verify that the parameters are not null
        if text is not None and context is not None and isinstance(emotions, list):

            # Create a string that lists the emotions in the array, separated by commas
            emotion_string = ", ".join(emotions)

             # Concatenate the text and context into a single string
            prompt="Text: " + text +  ". Sentiments:"+ emotion_string + ". Context:" +context 
            prompt=f'''You are a Ticket Response BOT. Please generate a formal, helpful, and empathetic response to the user’s message, taking into account their Emotions Detected and the relevant department, listed bellow. Make sure the response directly addresses the user’s issue and maintains a professional tone.

                Emotions detected: {', '.join(emotions)}
                Department: {context}
                User message: {text}
                User name: Default

                Response:'''
    
            generated_text = generate(prompt,model)

        return generated_text
    except ValueError as e:
        return str(e)

@app.route('/text_generation', methods=['POST'])
def translate_text():
    text = request.json['text']
    emotions = request.json['emotions']
    context = request.json['context']
    generatedText = generateText(text, emotions,context)
    return jsonify({'text': generatedText})
    


if __name__ == '__main__':
    app.run(host="localhost", port=8093, debug=True)
