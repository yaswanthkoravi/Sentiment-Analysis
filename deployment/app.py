from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as numpy
import pandas as pd
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
import joblib
import re
import nltk




import flask
app = Flask(__name__)


###################################################
pat1 = r'@[A-Za-z0-9_]+'        # remove @ mentions from tweets
pat2 = r'https?://[^ ]+'        # remove URLs from tweets
combined_pat = r'|'.join((pat1, pat2)) #addition of pat1 and pat2
www_pat = r'www.[^ ]+'         # remove URLs from tweets
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not", \
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",\
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",\
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",\
                "mustn't":"must not"}
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):  # define tweet_cleaner function to clean the tweets
    soup = BeautifulSoup(text, 'lxml')    # create beautiful soup object
    souped = soup.get_text()   # get only text from the tweets 
    
    bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed) # calling combined_pat
    stripped = re.sub(www_pat, '', stripped) #remove URLs
    lower_case = stripped.lower()      # converting all into lower case
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case) # converting words like isn't to is not
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)       # will replace # by space
    words = [x for x  in word_tokenize(letters_only) if len(x) > 1] # Word Punct Tokenize and only consider words whose length is greater than 1
    
    return (" ".join(l for l in words if l not in stopwords)).strip() # join the words

###################################################


@app.route('/')
def hello_world():
    return 'Welcome!!!! Go to index page to perform sentiment analysis'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('lr.pkl')
    tf_vect = joblib.load('tf_vect.pkl')
    to_predict_list = request.form.to_dict()
    # nltk.download('wordnet')
    clean_tweet_texts = tweet_cleaner(to_predict_list['review_text'])


    word_tokens = word_tokenize(clean_tweet_texts)

    df1 = [] # initialize list df1 to store words after lemmatization
    lemmatizer = WordNetLemmatizer() # create an object of WordNetLemmatizer
    for l in word_tokens: # for loop for every tokens in word_token
        df1.append(lemmatizer.lemmatize(l))
    a=" ".join(df1)

    pred = clf.predict(tf_vect.transform([a]))
    if pred[0]:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
