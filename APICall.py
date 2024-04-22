from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
import os

app = Flask(__name__,template_folder='../templates')

model_dir="C:/Users/navya/Desktop/ChatbotProject/Model"
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

@app.route('/')
def index():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    index_html_path = os.path.join(current_directory, 'templates', 'index.html')
    print("Location of index.html:", index_html_path)
    # Render the main interface where users can input their cravings or ingredients.
    return render_template('index.html')

@app.route('/get_recipe', methods=['POST'])
def get_recipe():
    try:
        user_input = request.json['user_input']
        # Tokenize the user input
        input_ids = tokenizer.encode(user_input, return_tensors='pt')
        # Generate a response from the model
        gpt_response = model.generate(
            input_ids,
            max_length=100,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        # Decode the generated response
        decoded_response = tokenizer.decode(gpt_response[0], skip_special_tokens=True)
        return jsonify({'response': decoded_response, 'status': 'success'})
    except Exception as e:
        # In case of an error, return an error message
        return jsonify({'response': str(e), 'status': 'error'})
    # Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 

