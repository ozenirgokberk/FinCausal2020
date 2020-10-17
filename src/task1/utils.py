import pandas as pd
import re

from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import cross_val_score
import nltk
import string

class Utils(object):

    def cleanText(self,text):
        review = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", str(text))
        review = re.sub(r"\([\s\S]*\)", " ", str(review))
        review = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", str(review))
        review = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", str(review))
        review = review.lower()
        review = re.sub(r"that's", "that is", str(review))
        review = re.sub(r"there's", "there is", str(review))
        review = re.sub(r"what's", "what is", str(review))
        review = re.sub(r"where's", "where is", str(review))
        review = re.sub(r"it's", "it is", str(review))
        review = re.sub(r"who's", "who is", str(review))
        review = re.sub(r"i'm", "i am", str(review))
        review = re.sub(r"she's", "she is", str(review))
        review = re.sub(r"he's", "he is", str(review))
        review = re.sub(r"they're", "they are", str(review))
        review = re.sub(r"who're", "who are", str(review))
        review = re.sub(r"ain't", "am not", str(review))
        review = re.sub(r"wouldn't", "would not", str(review))
        review = re.sub(r"shouldn't", "should not", str(review))
        review = re.sub(r"can't", "can not", str(review))
        review = re.sub(r"couldn't", "could not", str(review))
        review = re.sub(r"won't", "will not", str(review))
        review = re.sub(r" pm ", " ", str(review))
        review = re.sub(r" am ", " ", str(review))
        review = re.sub(r'[^\[\]]+(?=\])', " ", str(review))
        review = re.sub(r"\W", " ", str(review))
        review = re.sub(r"\d", " ", str(review))
        review = re.sub(r"\s+[a-z]\s+", " ", str(review))
        review = re.sub(r"\s+[a-z]$", " ", str(review))
        review = re.sub(r"^[a-z]\s+", " ", str(review))
        review = re.sub(r"\s+", " ", str(review))
        return review

    def remove_punc(self,text):
        table = str.maketrans("", "", string.punctuation)
        return text.translate(table)


    def getStopWords(self):
        return set(nltk.corpus.stopwords.words('english'))

    def remove_stops(self,text):
        text = [word.lower() for word in text.split() if word.lower() not in stop]
        return " ".join(text)


    def readData(self,path,inputColumnIndex=1,outputColumnIndex=2):
        df = pd.read_csv(path, error_bad_lines=False, sep=';')
        X = df.iloc[:, inputColumnIndex].values
        y = df.iloc[:, outputColumnIndex].values
        return X,y
    
    def draw_prediction_results(self,y_pred, y_test, my_tags, method):
        print('accuracy of ' + method + ': %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred, target_names=my_tags))
    
    def crossValidation(self,prediction,input,output,k=5):
        scores = cross_val_score(prediction, input,output, cv=k)
        print("Accuracy of Cross Validation Mean: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
