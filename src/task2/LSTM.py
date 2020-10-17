from sklearn.feature_extraction import DictVectorizer
from src.task2.utils import Task2Utils
from keras.models import Sequential
from keras.layers import Dense,LSTM

utils = Task2Utils()
df = utils.readDataAsDataFrame('fnpTask2.csv')
train,test = utils.getData(df)

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

# LSTM Model

model = Sequential()
model.add(LSTM((1), batch_input_shape=(None,5,1),return_sequences=True))
model.add(LSTM((1), return_sequences=False))
model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=1000,validation_data=(X_test,y_test))

results = model.predict(X_test)

loss, accuracy = model.evaluate(X, y_train_for_vectorizer, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test_for_vectorizer, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))