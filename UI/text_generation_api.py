
from flask import Flask, request, jsonify
import openai

app = Flask(__name__)
# Set up OpenAI API credentials and create a GPT-2 model instance~
API_KEY = "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"
openai.api_key = API_KEY
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


def generateText(text, emotions, product, sub_product, issue, sub_issue):
    try:

        #Verify that the parameters are not null
        if text is not None and product is not None and isinstance(emotions, list):
             # Concatenate the text and context into a single string
            prompt=f'''You are a Ticket Response BOT. Please generate a formal, helpful, and empathetic response to the user’s message, taking into account their Emotions Detected and the relevant department, listed bellow. Make sure the response directly addresses the user’s issue and maintains a professional tone.

                Emotions detected: {', '.join(emotions)}
                Issue: {issue}
                Sub_issue: {sub_issue}
                Product: {product}
                Sub_product: {sub_product}
                User message: {text}
                User name: User
                Response BOT Name: ShopAIHolic ST

                Response:'''
    
            generated_text = generate(prompt,model)

        return generated_text
    except ValueError as e:
        return str(e)

@app.route('/text_generation', methods=['POST'])
def translate_text():
    print("DENTRO DO translate_text")
    text = request.json['ticket_text_translated']
    emotions = request.json['emotions']
    product = request.json['product']
    sub_product = request.json['sub_product']
    issue = request.json['issue']
    sub_issue = request.json['sub_issue']

    # generatedText = generateText(text, emotions,product,sub_product, issue, sub_issue)
    generatedText = generateText(text, emotions, product, sub_product, issue, sub_issue) if API_KEY is not None  else "Ticket Response set to None. Not using API. To use API set your OpenAI Key."
    return jsonify({'ticket_answer': generatedText})
    


if __name__ == '__main__':
    app.run(host="localhost", port=8093, debug=True)
