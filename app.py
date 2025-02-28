import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, render_template, request
import requests

# Initialize the Flask app
app = Flask(__name__)

# Load GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2-medium"  # Medium for better text quality
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Function to generate responses using the GPT-2 model
def generate_response(prompt, model=model, tokenizer=tokenizer, max_length=600):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text from the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# Function to fetch academic references using the CrossRef API
def fetch_references(query, num_references=15):
    url = f"https://api.crossref.org/works?query={query}&rows={num_references}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        references = []
        for item in data['message']['items']:
            title = item.get('title', ['No title available'])[0]
            author = ", ".join([author.get('family', '') for author in item.get('author', [])])
            year = item.get('published', {}).get('date-parts', [[None]])[0][0]  # Year of publication
            journal = item.get('container-title', ['No journal or conference'])[0]
            url = item.get('URL', 'No URL available')
            references.append({"title": title, "author": author, "year": year, "journal": journal, "link": url})
        return references
    return []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        
        if question:
            # Generate responses for each chapter
            abstract_prompt = f"Abstract: {question}"
            introduction_prompt = f"Introduction: {question}"
            theoretical_background_prompt = f"Theoretical background: {question}"
            development_prompt = f"Development: {question}"
            tests_prompt = f"Tests: {question}"
            answer_to_question_prompt = f"Answer to the research question: {question}"
            conclusions_prompt = f"Conclusions: {question}"

            abstract = generate_response(abstract_prompt)
            introduction = generate_response(introduction_prompt)
            theoretical_background = generate_response(theoretical_background_prompt)
            development = generate_response(development_prompt)
            tests = generate_response(tests_prompt)
            answer_to_question = generate_response(answer_to_question_prompt)
            conclusions = generate_response(conclusions_prompt)

            # Fetch references
            references = fetch_references(question, num_references=15)

            return render_template('result.html', question=question, 
                                   abstract=abstract, introduction=introduction, 
                                   theoretical_background=theoretical_background,
                                   development=development, tests=tests,
                                   answer_to_question=answer_to_question, conclusions=conclusions,
                                   references=references)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3006)

