from flask import Flask,render_template,url_for,request
app = Flask(__name__)


from sklearn.externals import joblib
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
from os import sys
sys.setrecursionlimit(150000)

from gensim import corpora, models
import gensim


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    with open('lda.pkl','rb') as lda_model:
        lda_model = joblib.load(lda_model)


    with open('dictionary.pkl','rb') as dictionary:
        dictionary = joblib.load(dictionary)


    def clean(art):
        article = art.strip()
        tokens = RegexpTokenizer(r'\w+').tokenize(article.lower())
        tokens_clean = [token for token in tokens if token not in stopwords.words('english')]
        tokens_stemmed = [PorterStemmer().stem(token) for token in tokens_clean]
        return (tokens_stemmed)


    topics = lda_model.print_topics(num_topics=50, num_words=3)


    # input_text = str(input())
    # lis = clean(input_text.lower())


    # pred = list(lda_model[[dictionary.doc2bow(lis)]])
    # prediction = max(pred[0], key = lambda x: x[1])[0]
    # my_prediction = topics[prediction]

    if request.method == 'POST':
        message = request.form['message']
        data = str(message)
        lis = clean(data.lower())
        pred = list(lda_model[[dictionary.doc2bow(lis)]])
        prediction = max(pred[0], key = lambda x: x[1])[0]
        s = ' '.join(topics[prediction][1].split('*'))
        my_prediction = []
        my_prediction.append(s.split()[1].strip().replace('"', ''))
        my_prediction.append(s.split()[4].replace('"', ''))
        my_prediction.append(s.split()[7].replace('"', ''))
    return render_template('result.html', prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)