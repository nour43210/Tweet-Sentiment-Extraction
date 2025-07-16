import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify, send_file
from flask_bootstrap import Bootstrap  # Updated import
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import os
import time
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
bootstrap = Bootstrap(app)  # Updated initialization

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TweetSentimentAnalyzer:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(probability=True),
            'Random Forest': RandomForestClassifier()
        }
        self.current_model = 'Logistic Regression'
        self.model_trained = False
        self.training_history = []
        
        if dataset_path:
            self.load_data()
            self.preprocess_data()
            self.train_model()

    def load_data(self):
        """Load the dataset from the specified path"""
        try:
            # The dataset has no headers, so we'll add them
            self.df = pd.read_csv(self.dataset_path, encoding='latin-1', 
                                 header=None, names=['target', 'id', 'date', 'flag', 'user', 'text'])
            
            # Map target to sentiment (0=negative, 2=neutral, 4=positive)
            self.df['sentiment'] = self.df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
            
            # For this demo, we'll work with a smaller sample to speed up processing
            self.df = self.df.sample(frac=0.1, random_state=42)
            
        except Exception as e:
            raise Exception(f"Failed to load dataset: {str(e)}")

    def clean_text(self, text):
        """Preprocess and clean text data"""
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        text = re.sub(r'\@\w+|\#', '', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        text = text.lower()
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Lemmatization and remove stopwords
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split() 
                        if word not in self.stop_words])
        
        return text

    def preprocess_data(self):
        """Clean and prepare the data for modeling"""
        self.df['cleaned_text'] = self.df['text'].apply(self.clean_text)
        
        # Split data into features and target
        self.X = self.df['cleaned_text']
        self.y = self.df['target'].replace({4: 1, 0: 0})  # Convert to binary (0=negative, 1=positive)
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Vectorize the text data
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)

    def train_model(self, model_name=None):
        """Train the sentiment analysis model"""
        if model_name:
            self.current_model = model_name
            self.model = self.models[model_name]
        
        start_time = time.time()
        self.model.fit(self.X_train_vec, self.y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        train_preds = self.model.predict(self.X_train_vec)
        test_preds = self.model.predict(self.X_test_vec)
        
        train_acc = accuracy_score(self.y_train, train_preds)
        test_acc = accuracy_score(self.y_test, test_preds)
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': self.current_model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'training_time': training_time
        })
        
        self.model_trained = True
        
        return {
            'model': self.current_model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'training_time': training_time,
            'classification_report': classification_report(self.y_test, test_preds, output_dict=True)
        }

    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        if not self.model_trained:
            raise Exception("Model not trained yet. Please train the model first.")
            
        cleaned_text = self.clean_text(text)
        text_vec = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'cleaned_text': cleaned_text,
            'probability_positive': probability[1],
            'probability_negative': probability[0]
        }

    def generate_wordcloud(self, sentiment=None):
        """Generate word cloud for specific sentiment"""
        if sentiment:
            texts = ' '.join(self.df[self.df['sentiment'] == sentiment]['cleaned_text'])
        else:
            texts = ' '.join(self.df['cleaned_text'])
            
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texts)
        
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        
        return base64.b64encode(img.getvalue()).decode('utf-8')

    def save_model(self, filename):
        """Save the current model and vectorizer"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'model_name': self.current_model
            }, f)

    def load_model(self, filename):
        """Load a saved model and vectorizer"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.current_model = data['model_name']
            self.model_trained = True

# Initialize analyzer (will be loaded with dataset when needed)
analyzer = TweetSentimentAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                try:
                    global analyzer
                    analyzer = TweetSentimentAnalyzer(filepath)
                    analyzer.preprocess_data()
                    training_result = analyzer.train_model(request.form.get('model_type', 'Logistic Regression'))
                    return render_template('index.html', 
                                        training_result=training_result,
                                        model_trained=True,
                                        models=list(analyzer.models.keys()),
                                        current_model=analyzer.current_model,
                                        wordcloud_pos=analyzer.generate_wordcloud('positive'),
                                        wordcloud_neg=analyzer.generate_wordcloud('negative'))
                except Exception as e:
                    return render_template('index.html', error=str(e))
        
        elif 'text' in request.form:
            text = request.form['text']
            try:
                result = analyzer.predict_sentiment(text)
                return render_template('index.html', 
                                    prediction_result=result,
                                    model_trained=analyzer.model_trained,
                                    models=list(analyzer.models.keys()),
                                    current_model=analyzer.current_model)
            except Exception as e:
                return render_template('index.html', error=str(e), model_trained=analyzer.model_trained)
    
    return render_template('index.html', 
                          models=list(analyzer.models.keys()),
                          model_trained=analyzer.model_trained)

@app.route('/train', methods=['POST'])
def train_model():
    model_type = request.form.get('model_type', 'Logistic Regression')
    try:
        training_result = analyzer.train_model(model_type)
        return jsonify({
            'success': True,
            'result': training_result,
            'wordcloud_pos': analyzer.generate_wordcloud('positive'),
            'wordcloud_neg': analyzer.generate_wordcloud('negative')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_model', methods=['POST'])
def save_model():
    filename = request.form.get('filename', 'sentiment_model.pkl')
    try:
        analyzer.save_model(filename)
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_model', methods=['POST'])
def load_model():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            try:
                analyzer.load_model(file)
                return jsonify({
                    'success': True,
                    'model_name': analyzer.current_model,
                    'wordcloud_pos': analyzer.generate_wordcloud('positive'),
                    'wordcloud_neg': analyzer.generate_wordcloud('negative')
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': False, 'error': 'No file provided'})

@app.route('/history')
def get_history():
    return jsonify(analyzer.training_history)

@app.route('/templates/index.html')
def serve_template():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)