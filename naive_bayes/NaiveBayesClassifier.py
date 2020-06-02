import pandas as pd
import numpy as np
import math
from scipy.stats import norm


class NaiveBayesClassifier:
    def __init__(self, target_class, proba_type='log', pseudo_count=0):
        self.target_class = target_class
        self.pseudo_count = pseudo_count
        self.prediction_func = self.point_predict_proba_log if proba_type == 'log' else self.point_predict_proba
        self.model = {}

    def fit(self, x, y):
        apriori = y.value_counts(normalize=True)
        self.model['apriori'] = apriori

        for col in x.columns:
            if x[col].dtype.kind in 'biufc':
                self.model[col] = {
                    "numeric": x[col].dtype.kind in 'biufc',
                }
                for class_value in list(self.model['apriori'].keys()):
                    class_value_data = x[y == class_value]
                    self.model[col][class_value] = {
                        "mean": np.mean(class_value_data[col]),
                        "std": np.std(class_value_data[col])
                    }
            else:
                cross_matrix = pd.crosstab(x[col], y)
                self.model[col] = {
                    "numeric": x[col].dtype.kind in 'biufc',
                    "cross_matrix": cross_matrix
                }
        return self.model

    @staticmethod
    def pdf_value(x, mean, std):
        return norm.pdf(x, loc=mean, scale=std)
        # return (1/(np.sqrt(2)*math.pi))*np.exp(-((x-mean)**2)/(2*std**2))

    @staticmethod
    def conditional_proba(column_matrix, class_value, obs_column_value, pseudo_count=0):
        return (column_matrix[class_value][obs_column_value] + pseudo_count) / (column_matrix[class_value].sum() + pseudo_count * column_matrix.shape[0])

    def point_predict_proba_log(self, point):
        prediction = {}
        for class_value in self.model['apriori'].index:
            cumulative_proba = 0
            for atr in self.model:
                if atr == 'apriori':
                    cumulative_proba = cumulative_proba + np.log(self.model['apriori'][class_value])
                else:
                    if self.model[atr]['numeric']:
                        item_proba = self.pdf_value(point[atr], self.model[atr][class_value]['mean'], self.model[atr][class_value]['std'])
                    else:
                        item_proba = self.conditional_proba(self.model[atr]['cross_matrix'], class_value, point[atr], self.pseudo_count)
                    cumulative_proba = cumulative_proba + np.log(item_proba)
            prediction[class_value] = cumulative_proba
        return prediction

    def point_predict_proba(self, point):
        prediction = {}
        for class_value in self.model['apriori'].index:
            cumulative_proba = 1
            for atr in self.model:
                if atr == 'apriori':
                    cumulative_proba = cumulative_proba * self.model['apriori'][class_value]
                else:
                    if self.model[atr]['numeric']:
                        item_proba = self.pdf_value(point[atr], self.model[atr][class_value]['mean'], self.model[atr][class_value]['std'])
                    else:
                        item_proba = self.conditional_proba(self.model[atr]['cross_matrix'], class_value, point[atr], self.pseudo_count)
                    cumulative_proba = cumulative_proba * item_proba
            prediction[class_value] = cumulative_proba
        return prediction

    def point_predict(self, point_row):
        prediction_proba = self.prediction_func(point_row)
        point_row['prediction'] = max(prediction_proba, key=lambda x: prediction[x])
        for value in data[target_class].unique():
            data_new.loc[i, f"{target_class}={value}"] = prediction_proba[value]
        return point_row

    def predict(self, x):
        x_ = x.copy()
        for i in range(len(x_)):
            point = x_.loc[i]
            prediction = self.prediction_func(point)
            x_.loc[i, 'prediction'] = max(prediction, key=lambda x: prediction[x])
            for class_value in y.unique():
                x_.loc[i, f"{target_class}={class_value}"] = prediction[class_value]
        return x_


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    data = pd.read_csv('../data/drug.csv')
    target_class = 'Drug'
    # data = pd.read_csv('../data/prehlada.csv')
    # target_class = 'Prehlada'

    x = data.drop(target_class, axis=1)
    y = data[target_class]

    model_nb = NaiveBayesClassifier(target_class, 'log', 1e-3)
    model_nb.fit(x, y)
    data_new = model_nb.predict(data)
    print(data_new)

    data_new['matched'] = data_new[target_class] == data_new['prediction']
    print(f"Share of matched: {data_new['matched'].sum() / data_new.shape[0]}")


