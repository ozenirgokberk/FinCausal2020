from src.task1.utils import Utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
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

linear_svm = Pipeline([('vect', CountVectorizer(min_df=1, max_df=0.5, analyzer='word', ngram_range=(1, 3)
                                                )),
                       ('tfidf', TfidfTransformer()),
                       ('linear_svc',
                        SVC(kernel='linear'))
                       ])
linear_svm.fit(X_train, y_train)
y_pred_svc = linear_svm.predict(X_test)


# other usages for SVMs

# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train)
# tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)

# linear_svc = SGDClassifier(loss='hinge', penalty='l2',
#                            alpha=1e-3, random_state=42,
#                            max_iter=5, tol=None)

# linear_svc.fit(X_train, y_train)
# y_pred_svc = linear_svc.predict(X_test)







# prediction results

utils.draw_prediction_results(y_pred_svc,y_test,ourTags,"Naive Bayes")

# 5-fold cross validation results
utils.crossValidation(X,y,k=5)
