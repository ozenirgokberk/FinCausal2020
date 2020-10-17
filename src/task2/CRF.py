from src.task2.utils import Task2Utils
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats

utils = Task2Utils()
df = utils.readDataAsDataFrame('fnpTask2.csv')
train, test = utils.getData(df)

X_train, y_train, _ = utils.preprocessData(train)
X_test, y_test, tokens_test = utils.preprocessData(test)

crf = sklearn_crfsuite.CRF(
    algorithm='l2sgd',
    min_freq=1.0,
    max_iterations=1000,
    all_possible_states=True,
    all_possible_transitions=True,
    c2=0.01
)
params_space = {
    'c2': scipy.stats.expon(scale=0.05)
}
f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=['_', 'C', 'E'])
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)

rs.fit(X_train, y_train)
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
crf = rs.best_estimator_

# Passive Aggressive Algorithm usage

# crf = sklearn_crfsuite.CRF(
#     algorithm='pa',
#     min_freq=1.0,
#     max_iterations=100,
#     all_possible_states=True,
#     all_possible_transitions=False
# )

# Averaged Perceptron Algorithm usage

# crf = sklearn_crfsuite.CRF(
#     algorithm='ap',
#     min_freq=1.0,
#     max_iterations=100,
#     all_possible_states=True,
#     all_possible_transitions=False
# )


crf.fit(X_train, y_train)
y_pred = crf.predict(X_test)
tags = ['_', 'C', 'E']
utils.printResult(y_test, y_pred, tags, digits=3)
