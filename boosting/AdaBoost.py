import math
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:

    def __init__(self, ensemble_size=5, learning_rate=1/2, base_model=[GaussianNB]):
        self.learning_rate = learning_rate
        self.ensemble_size = ensemble_size
        self.model_weights = np.zeros(ensemble_size)
        self.ensembles = []
        self.base_models = base_model if base_model else [GaussianNB]

    def fit(self, x, y, log=True):
        n, m = x.shape

        instance_weights = pd.Series(np.array([1/n]*n), index=data.index)
        self.model_weights = np.zeros(self.ensemble_size)
        self.ensembles = []

        for i in range(self.ensemble_size):

            model = random.choice(self.base_models)()
            model = model.fit(x, y, sample_weight=instance_weights)
            error = (model.predict(x)-y).abs()
            total_error = (error*instance_weights).sum()
            if total_error != 0:
                base_model_weight = self.learning_rate * math.log((1-total_error)/total_error)
            else:
                base_model_weight = 1.0

            self.ensembles.append(model)
            self.model_weights[i] = base_model_weight

            instance_weights = instance_weights * np.exp(-base_model_weight*(error*2-1)*(-1))
            instance_weights = instance_weights/instance_weights.sum()

            if log:
                predictions = pd.DataFrame([model.predict(x) for model in self.ensembles]).T
                predictions = np.sign(predictions.dot(self.model_weights[:i+1]))
                print(f"{i+1}. model: {type(model).__name__}")
                print(f"Ensemble with {i+1} models, accuracy: {accuracy_score(y, predictions)}")

    def predict(self, x):
        predictions = pd.DataFrame([model.predict(x) for model in self.ensembles]).T
        return np.sign(predictions.dot(self.model_weights))


if __name__ == "__main__":
    data = pd.read_csv('../data/drugY.csv')
    x = data.drop('Drug', axis=1)
    y = data['Drug']*2-1
    n, m = x.shape

    x = pd.get_dummies(x)
    # model = AdaBoost()
    model = AdaBoost(learning_rate=0.1, ensemble_size=10, base_model=[GaussianNB, DecisionTreeClassifier])
    model.fit(x, y)
    data['predicted'] = model.predict(x)
    print(data.head())
    print('Final ensemble accuracy: ', accuracy_score(y, data.predicted))
