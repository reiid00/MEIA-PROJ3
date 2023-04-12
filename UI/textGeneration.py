
from flask import Flask, request, jsonify
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)


def load_generation_model():
    # Load the pre-trained GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer, model


def generate(prompt):
    if model is None:
        # Load the translation model
        tokenizer, model = load_generation_model()

   # Create a pipeline for text generation
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate a response
    generated_text = text_generator(prompt, max_length=200, num_return_sequences=1)

    # Extract the response text
    response = generated_text[0]['generated_text'][len(prompt):].strip()

    return response


# Set the context, emotions, and prompt text
context = "IT support"
emotions = ["frustrated"]
prompt_text = "Generate response to client about the ticket that he created about not being able to access his account"



def generateText(text, emotions, context):
    try:
     #Verify that the parameters are not null
        if text is not None and context is not None and isinstance(emotions, list):

            # Create a string that lists the emotions in the array, separated by commas
            emotion_string = ", ".join(emotions)

            print(emotion_string)

             # Concatenate the text and context into a single string
            prompt="Text: " + text +  ". Sentiments:"+ emotion_string + ". Context:" +context 
    
            generated_text = generate(prompt)

        return generated_text
    except ValueError as e:
        return str(e)

@app.route('/translate_text', methods=['POST'])
def translate_text():
    text = request.json['text']
    emotions = request.json['emotions']
    context = request.json['context']
    generatedText = generateText(text, emotions,context)
    return jsonify({'text': generatedText})
    


if __name__ == '__main__':
    app.run(host="localhost", port=8091, debug=True)
