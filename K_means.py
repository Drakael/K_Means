import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


class K_Means:
    def __init__(self, vector_list, C, iterations=100):
        self.vector_list = vector_list
        self.vec_dim = len(vector_list[0])
        self.C = C
        self.iterations = iterations
        self.centroids = list()
        min_ = np.min(vector_list)
        max_ = np.max(vector_list)
        self.centroids = np.random.uniform(min_, max_, (C, self.vec_dim))
        print('centroids', self.centroids)
        self.elem2cluster = list()
        self.cost = None

    def __associate_to_clusters(self):
        self.cost = np.zeros((self.C, ))
        self.elem2cluster = list()
        for elem in self.vector_list:
            closest_centroid = None
            closest_centroid_distance = None
            for c, centroid in enumerate(self.centroids):
                distance = euclidean(elem, centroid)
                if closest_centroid_distance is None:
                    closest_centroid_distance = distance
                    closest_centroid = c
                elif closest_centroid_distance > distance:
                    closest_centroid_distance = distance
                    closest_centroid = c
            self.elem2cluster.append(closest_centroid)
            self.cost[closest_centroid] += closest_centroid_distance

    def __update_centroids(self):
        for c, centroid in enumerate(self.centroids):
            sum_ = np.zeros((self.vec_dim, ))
            nb_elems = 0
            for e, cluster_id in enumerate(self.elem2cluster):
                if cluster_id == c:
                    sum_ += self.vector_list[e]
                    nb_elems += 1
            self.centroids[c] = sum_ / nb_elems

    def iterate(self):
        self.__associate_to_clusters()
        for i in range(self.iterations):
            print('iterations', i, 'cost', np.sum(self.cost))
            self.__update_centroids()
            self.__associate_to_clusters()
            #     colors = ['r', 'g', 'b']
            #     color_list = [colors[cluster] for cluster in self.elem2cluster]
            #     plt.figure()
            #     plt.scatter(self.vector_list[:, 0], self.vector_list[:, 1], color=color_list)
            #     plt.show()

if __name__ == '__main__':
    data = np.random.uniform(0, 10, (130, 2))
    kmean = K_Means(data, 5, 300)
    kmean.iterate()
    print('data', data)
    print('kmean.centroids', kmean.centroids)
    print('kmean.elem2cluster', kmean.elem2cluster)
    colors = ['g', 'b', 'c', 'm', 'y']
    color_list = [colors[cluster] for cluster in kmean.elem2cluster]
    plt.figure()
    plt.scatter(kmean.centroids[:, 0], kmean.centroids[:, 1],
                color='r', marker='x')
    plt.scatter(data[:, 0], data[:, 1], color=color_list)
    plt.show()
