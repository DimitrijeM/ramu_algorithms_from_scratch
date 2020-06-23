import sys

import numpy as np
import pandas as pd


class KMeansClustering:
    def __init__(self, k, distance='euclidean', retry_num=5, init_centroids='random'):
        self.k = k
        self.retry_num = retry_num
        self.x_mean, self.x_std = 0, 1
        self.model = {}
        self.distance_measure = KMeansClustering.euclidean_distance if distance == 'euclidean' else KMeansClustering.city_block_distance
        self.init_centroids = KMeansClustering.select_farest_centroids if init_centroids == 'farest' else KMeansClustering.select_random_centroids

    @staticmethod
    def get_np_array(x):
        if type(x).__module__ != np.__name__:
            return x.values
        else:
            return x

    @staticmethod
    def euclidean_distance(point, centroids):
        dist = (point-centroids)
        return (dist**2).sum(axis=1)

    @staticmethod
    def city_block_distance(point, centroids):
        return abs(point-centroids).sum(axis=1)

    def normalize(self, x):
        return (x-self.x_mean)/self.x_std

    @staticmethod
    def select_farest_centroids(x, k):
        n, m = x.shape
        dists = np.zeros(shape=(n, k))
        init_centroids = x[np.random.randint(n, size=1), :]
        for cluster_num in range(k-1):
            dists[:, cluster_num] = ((x - init_centroids[cluster_num])**2).sum(axis=1)
            init_centroids = np.concatenate((init_centroids, x[np.argmax(dists[:, cluster_num]), :].reshape(1, -1)), axis=0)
        return init_centroids

    @staticmethod
    def select_random_centroids(x, k):
        n, m = x.shape
        return x[np.random.randint(n, size=k), :]

    def fit(self, x, attribute_weights=None, max_iter=3, log=True):
        x = KMeansClustering.get_np_array(x)
        n, m = x.shape

        if attribute_weights is None:
            attribute_weights = np.ones(m)

        result_centroids_dict = {}
        result_qualities = []

        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)
        x = (x-self.x_mean)/self.x_std

        for retry in range(self.retry_num):
            centroids = self.init_centroids(x, self.k)

            assign = np.zeros(n)
            old_quality = np.array(float('inf'))
            for it in range(max_iter):
                quality = np.full(self.k, sys.maxsize/2)
                for j in range(n):
                    point = x[j, :]
                    dist = self.distance_measure(point, centroids)
                    assign[j] = np.argmin(dist)

                for c in range(self.k):
                    subset = x[assign == c]
                    if subset.shape[0] > 0:
                        centroids[c] = subset.mean(axis=0)
                        quality[c] = (subset.var(axis=0) * attribute_weights).sum() * subset.shape[0]
                    else:
                        quality[c] = sys.maxsize/2

                if (abs(old_quality-quality) < 0.1).all():
                    break
                old_quality = quality

            if log:
                print(f"Retry={retry}:")
                print(f"Quality: {quality.sum()}, {quality}")
                print(f"Centroids: {centroids}")

            result_centroids_dict['model_'+str(retry)] = centroids
            result_qualities += [quality.sum()]

        best_index = np.argmin(result_qualities)
        print(f"best quality: {best_index}, quality: {result_qualities[best_index]}")

        self.model = result_centroids_dict['model_'+str(best_index)]
        print('model: ' + str(self.model))
        return self.model

    def predict(self, x):
        x = KMeansClustering.get_np_array(x)
        x = self.normalize(x)
        n, m = x.shape
        assign = np.zeros(n)
        for j in range(n):
            point = x[j, :]
            dist = self.distance_measure(point, self.model)
            assign[j] = np.argmin(dist)
        return assign

    @staticmethod
    def calculate_point_distance_from_group(point, group):
        return KMeansClustering.euclidean_distance(point, group).sum()

    @staticmethod
    def silhouette_score(x, cluster):
        k = np.unique(cluster)

        silhouette = np.zeros(shape=k.shape[0])
        for i in range(k.shape[0]):
            i_cluster_data = x[cluster == k[i]]
            if i_cluster_data.shape[0] == 1:
                silhouette[i] = 0
            else:
                mean_distance_in_belonging = KMeansClustering.calculate_point_distance_from_group(i_cluster_data,
                                                                                                  i_cluster_data.mean(axis=0)) / (i_cluster_data.shape[0]-1)
                mean_distance_i_from_j = np.zeros(shape=k.shape[0])
                for j in range(k.shape[0]):
                    mean_distance_i_from_j[j] = KMeansClustering.calculate_point_distance_from_group(i_cluster_data,
                                                                                                     x[cluster == k[j]].mean(axis=0)) / (i_cluster_data.shape[0])

                smallest_distance_to_neighbor = np.min(mean_distance_i_from_j)
                silhouette[i] = np.abs(smallest_distance_to_neighbor - mean_distance_in_belonging) / \
                                np.max([mean_distance_in_belonging, smallest_distance_to_neighbor])
        return silhouette.mean()

    @staticmethod
    def select_best_k(x, k_range=range(2, 8)):
        x = KMeansClustering.get_np_array(x)
        silhouette_scores = np.zeros(shape=len(k_range))
        for i in range(len(k_range)):
            model = KMeansClustering(k_range[i], 'euclidean', 1, 'random')
            model.fit(x)
            cluster = model.predict(x)
            silhouette_scores[i] = KMeansClustering.silhouette_score(x, cluster)
            print(silhouette_scores)
        the_best_k = k_range[np.argmin(silhouette_scores)]
        print(f"The best k={the_best_k} with obtained score={silhouette_scores.min()}")
        return the_best_k


if __name__ == "__main__":

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # data = pd.read_csv('../data/life.csv').set_index('country')
    data = pd.read_csv('../data/boston.csv')

    # k = 2
    k = KMeansClustering.select_best_k(data, k_range=range(2, 10))

    # model = KMeansClustering(k, 'euclidean', 5, 'random')
    model = KMeansClustering(k, 'city-block', 5, 'farest')

    # centroids = model.fit(data)

    # attribute_weights = [2, 0.5, 0.5, 1, 1]
    attribute_weights = np.random.random(size=14)
    # attribute_weights = np.zeros(14)
    # attribute_weights[-1] = 1
    print(f"Attribute weights: {attribute_weights}")

    centroids = model.fit(data, attribute_weights)

    data['cluster'] = model.predict(data)
    print(data)
    print(data.cluster.value_counts())

