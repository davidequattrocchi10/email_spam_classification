import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    '''
    Remove punctuation, numbers and convert text to lowercase.
    '''
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    return text.lower()   # Convert text into lowercase

def tokenize_text(text):
    '''
    Split the text in words
    '''
    return text.split()

def remove_stopwords(tokens):
    '''
    Remove the stopwords from a list of token
    '''
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    '''
    Reduce tokens in their base form
    '''
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    '''
    Apply all the functions for preprocessing
    '''
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)
