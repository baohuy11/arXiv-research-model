import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class DataPreprocessor:
    def __init__(self, data_path, random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.preprocessor = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        try:
            # Convert to lowercase
            text = str(text).lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Warning: Error processing text: {str(e)}")
            return text  # Return original text if processing fails
        
    def extract_text_features(self, text):
        """Extract additional features from text"""
        word_count = len(text.split())
        char_count = len(text)
        avg_word_length = char_count / (word_count + 1)  # Add 1 to avoid division by zero
        
        # Add more text features
        sentences = text.split('.')
        sentence_count = len(sentences)
        avg_sentence_length = word_count / (sentence_count + 1)
        
        # Count unique words
        unique_words = len(set(text.split()))
        lexical_diversity = unique_words / (word_count + 1)
        
        return pd.Series({
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'unique_words': unique_words,
            'lexical_diversity': lexical_diversity
        })
        
    def load_data(self):
        """Load and perform initial data preprocessing"""
        print("Loading data...")
        data = pd.read_csv(self.data_path)
        
        # Get top 10 categories
        top_categories = data['category'].value_counts().nlargest(10).index
        data['category_code'] = data['category'].apply(lambda x: x if x in top_categories else 'other')
        
        # Process text data
        print("Processing text data...")
        tqdm.pandas(desc="Processing summaries")
        data['processed_summary'] = data['summary'].progress_apply(self.preprocess_text)
        data['processed_title'] = data['title'].progress_apply(self.preprocess_text)
        
        # Extract text features
        print("Extracting text features...")
        tqdm.pandas(desc="Extracting features")
        text_features = data['summary'].progress_apply(self.extract_text_features)
        data = pd.concat([data, text_features], axis=1)
        
        # Process authors column to get count and features
        data['authors'] = data['authors'].apply(eval)
        data['authors_count'] = data['authors'].apply(len)
        
        # Process dates
        data['published_date'] = pd.to_datetime(data['published_date'])
        data['updated_date'] = pd.to_datetime(data['updated_date'])
        data['revision_time_days'] = (data['updated_date'] - data['published_date']).dt.total_seconds() / (24*60*60)
        
        # Extract month and year features
        data['pub_month'] = data['published_date'].dt.month
        data['pub_year'] = data['published_date'].dt.year
        
        return data
        
    def random_sample_per_class(self, df, class_column, fraction):
        """Perform stratified random sampling per class with minimum samples"""
        min_samples = 100  # Minimum samples per class
        
        sampled_df = df.groupby(class_column).apply(
            lambda x: x.sample(n=max(min_samples, int(len(x) * fraction)), 
                               random_state=self.random_state, replace=len(x) < min_samples)
        ).reset_index(drop=True)
        
        return sampled_df
        
    def create_preprocessor(self):
        """Create preprocessing pipeline for numerical and text features"""
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]
        )

        summary_transformer = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=500,
                min_df=5,
                max_df=0.85,
                ngram_range=(1, 2),
                stop_words='english'
            ))
        ])

        title_transformer = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=100,
                min_df=2,
                max_df=0.9,
                stop_words='english'
            ))
        ])

        category_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("summary", summary_transformer, "processed_summary"),
                ("title", title_transformer, "processed_title"),
                ("category", category_transformer, ["category_code"]),
                ("num", numeric_transformer, [
                    'authors_count', 'revision_time_days', 
                    'word_count', 'char_count', 'avg_word_length',
                    'sentence_count', 'avg_sentence_length',
                    'unique_words', 'lexical_diversity'
                ]),
            ],
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
        
    def prepare_data(self, sample_fraction=0.1, test_size=0.1):
        """Main method to prepare data for modeling"""
        # Load and preprocess data
        data = self.load_data()
        
        # Sample data
        sampled_df = self.random_sample_per_class(data, 'category_code', sample_fraction)
        
        # Define features
        features = [
            'processed_summary', 'processed_title', 'category_code',
            'authors_count', 'revision_time_days',
            'word_count', 'char_count', 'avg_word_length',
            'sentence_count', 'avg_sentence_length',
            'unique_words', 'lexical_diversity'
        ]
        
        X = sampled_df[features]
        y = sampled_df['category_code']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, 
            shuffle=True, random_state=self.random_state
        )
        
        # Create preprocessor if not already created
        if self.preprocessor is None:
            self.create_preprocessor()
            
        return X_train, X_test, y_train, y_test
