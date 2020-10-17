from src.task1.utils import Utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

utils = Utils()
X,y=utils.readData('fnpTask1.csv')

ourTags =['0','1']
corpus=[]

for i in range(0, len(X)):
    t = utils.clean_text(X[i])
    t = utils.remove_punc(t)
    t = utils.remove_stops(t)
    corpus.append(t)


X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.3, random_state=0)

lrp = Pipeline([('vect', CountVectorizer(min_df=0.2, max_df=3, analyzer='word', ngram_range=(1, 3),
                                         )),
                ('tfidf', TfidfTransformer()),
                ('lr', LogisticRegression())
                ])
lrp.fit(X_train, y_train)

y_pred = lrp.predict(X_test)

# prediction results

utils.draw_prediction_results(y_pred,y_test,ourTags,"logistic regression")

# 5-fold cross validation results
utils.crossValidation(X,y,k=5)
