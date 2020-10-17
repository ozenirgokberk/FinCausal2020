from sklearn.feature_extraction import DictVectorizer
from src.task2.utils import Task2Utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn_crfsuite import metrics

utils = Task2Utils()
df = utils.readDataAsDataFrame('fnpTask2.csv')
train, test = utils.getData(df)

X_train, y_train, _ = utils.preprocessData(train)
X_test, y_test, tokens_test = utils.preprocessData(test)

X_train_for_vectorizer = []
y_train_for_vectorizer = []

X_test_for_vectorizer = []
y_test_for_vectorizer = []

for i in X_train:
    for t in i:
        X_train_for_vectorizer.append(t)

for i in X_test:
    for t in i:
        X_test_for_vectorizer.append(t)

for i in y_train:
    for t in i:
        if t == 'E':
            y_train_for_vectorizer.append(1)
        if t == 'C':
            y_train_for_vectorizer.append(2)
        if t == '_':
            y_train_for_vectorizer.append(0)

for i in y_test:
    for t in i:
        if t == 'E':
            y_test_for_vectorizer.append(1)
        if t == 'C':
            y_test_for_vectorizer.append(2)
        if t == '_':
            y_test_for_vectorizer.append(0)

vectorizer = DictVectorizer(sparse=False)
X = vectorizer.fit_transform(X_train_for_vectorizer)
X_test = vectorizer.transform(X_test_for_vectorizer)

classifier = LogisticRegression()
classifier.fit(X, y_train_for_vectorizer)

y_pred = classifier.predict(X_test)

scores_dt = cross_val_score(classifier, X, y_train_for_vectorizer, cv=5)
print(classification_report(y_test_for_vectorizer, y_pred))
