from flask import Flask, render_template, request, redirect, url_for
import json
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')

app = Flask(__name__)

# Load schemes data from JSON file
with open('schemes_updated.json', 'r') as file:
    schemes_data = json.load(file)

# Initialize Porter Stemmer
porter = PorterStemmer()

# Preprocess and tokenize text
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    stemmed_tokens = [porter.stem(word) for word in tokens]  # Apply Porter Stemmer
    return ' '.join(stemmed_tokens)

# Preprocess scheme data
for scheme in schemes_data:
    scheme['processed_details'] = preprocess_text(scheme.get('details', ''))
    scheme['processed_query'] = preprocess_text(scheme['scheme_name'])  # Assuming scheme name is the query
    scheme['processed_gender'] = preprocess_text(scheme.get('gender', ''))
    scheme['processed_caste'] = preprocess_text(scheme.get('caste', ''))
    scheme['processed_is_student'] = preprocess_text(scheme.get('isStudent', ''))
    scheme['processed_state'] = preprocess_text(scheme.get('beneficiaryState', ''))

# Extract details, query, gender, caste, isStudent, and state
scheme_details = [scheme['processed_details'] for scheme in schemes_data]
scheme_queries = [scheme['processed_query'] for scheme in schemes_data]
scheme_genders = [scheme['processed_gender'] for scheme in schemes_data]
scheme_castes = [scheme['processed_caste'] for scheme in schemes_data]
scheme_is_students = [scheme['processed_is_student'] for scheme in schemes_data]
scheme_states = [scheme['processed_state'] for scheme in schemes_data]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(scheme_details + scheme_queries)

# Function to recommend schemes based on user query, gender, caste, isStudent, and state
def recommend_schemes(query, gender='All', caste='All', is_student='All', state='All', top_n=15):
    # Preprocess user query, gender, caste, isStudent, and state
    processed_query = preprocess_text(query)
    processed_gender = preprocess_text(gender)
    processed_caste = preprocess_text(caste)
    processed_is_student = preprocess_text(is_student)
    processed_state = preprocess_text(state)
    
    # Filter schemes based on gender, caste, isStudent, and state preferences
    filtered_schemes = [scheme for scheme, gender_val, caste_val, is_stud, state_val in zip(schemes_data, scheme_genders, scheme_castes, scheme_is_students, scheme_states)
                        if processed_gender in ['all', '', gender_val]
                        and processed_caste in ['all', '', caste_val]
                        and processed_is_student in ['all', '', is_stud]
                        and processed_state in ['all', '', state_val]]
    if not filtered_schemes:
        print("No schemes found for the specified gender, caste, isStudent, and state preferences.")
        return []
    
    # Calculate cosine similarity between query vector and scheme vectors for filtered schemes
    cosine_similarities = cosine_similarity(tfidf_vectorizer.transform([processed_query]), tfidf_vectorizer.transform([s['processed_details'] for s in filtered_schemes]))
    
    # Get indices of top N similar schemes
    top_indices = cosine_similarities.argsort()[0][-top_n:][::-1]
    
    # Return top N recommended schemes
    recommended_schemes = [filtered_schemes[idx]['sr_no'] for idx in top_indices]
    return recommended_schemes

# Function to get scheme data by sr_no from schemes.json
def get_data_by_sr_no(sr_no, data):
    for item in data:
        if item["sr_no"] == sr_no:
            return item
    return None

# Load JSON data from the file
with open("schemes.json", "r") as file:
    data_dict = json.load(file)

@app.route('/')
def index():
    return render_template('index.html', recommendations=None)

@app.route('/main', methods=['POST'])
def main():
    # Render the main.html template
    return render_template('main.html', recommendations=None)


@app.route('/recommend', methods=['POST'])
def recommend():
    user_query = request.form['query']
    user_gender = request.form['gender']
    user_caste = request.form['caste']
    user_state = request.form['state']
    user_student = request.form['student']

    recommended = recommend_schemes(user_query, user_gender, user_caste, user_student, user_state)

    recommendations = []
    counter = 1
    for sr_no in recommended:
        result = get_data_by_sr_no(sr_no, data_dict)
        if result:
            recommendations.append((counter, result['scheme_name']))
            counter += 1
        else:
            recommendations.append((counter, f'Data for sr_no {sr_no} not found.'))
            counter += 1

    return render_template('main.html', recommendations=recommendations)

@app.route('/details/<scheme_name>')
def details(scheme_name):
    # Load scheme data from the scheme.json file
    with open('schemes.json', 'r') as file:
        scheme_data = json.load(file)
    
    # Look for the scheme data based on the scheme name
    scheme = next((s for s in scheme_data if s['scheme_name'] == scheme_name), None)
    if scheme:
        return render_template('details.html', scheme_data=scheme)
    else:
        return "Scheme not found."
    
    
if __name__ == '__main__':
    app.run(debug=True)

