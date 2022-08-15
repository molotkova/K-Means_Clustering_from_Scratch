import numpy as np
from sklearn.datasets import load_wine
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

## Premade

def plot_comparison(data, predicted_clusters, true_clusters=None, centers=None, show=True):

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()

## Student

class CustomKMeans:
    def __init__(self, k):
        self.k = k

    def find_nearest_center(self, data: np.ndarray, centers: np.ndarray):
        return np.argmin(cdist(data, centers), axis=1)

    def calculate_new_centers(self, data: np.ndarray, data_labels: np.ndarray):
        new_centers = []
        for i in range(len(np.unique(data_labels))):
            new_centers.append(data[data_labels == i].mean(axis=0))
        return np.array(new_centers)

    def fit(self, data: np.ndarray, eps=1e-6) -> float:
        centers = data[:self.k]

        while True:
            labels = self.find_nearest_center(data, centers)
            new_centers = self.calculate_new_centers(data, labels)
            if np.linalg.norm(centers - new_centers) < eps:
                break
            else:
                centers = new_centers

        self.centers = centers

        result_squares = 0
        labels = self.find_nearest_center(data, centers)

        for i in range(self.k):
            result_squares += np.linalg.norm(data[labels == i] - centers[i])

        return result_squares / self.k

    def predict(self, data: np.ndarray):
        return self.find_nearest_center(data, self.centers)


def find_appropriate_k(data, limit):
    kmeans = CustomKMeans(1)
    error = kmeans.fit(data)
    k = 2
    while True:
        kmeans = CustomKMeans(k)
        new_error = kmeans.fit(data)
        if (error - new_error) / error < limit:
            return k - 1
        else:
            error = new_error
            k += 1

## Premade

if __name__ == '__main__':
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    X_full = X_full.values
    y_full = y_full.values

    scaler = MinMaxScaler()
    X_full = scaler.fit_transform(X_full)

    optimal_k = find_appropriate_k(X_full, 0.2)

    kmeans = CustomKMeans(optimal_k)
    kmeans.fit(X_full)
    predicted_labels = kmeans.predict(X_full)

    plot_comparison(X_full, predicted_labels, true_clusters=y_full, centers=kmeans.centers,
                    show=False)

    print(list(predicted_labels[:20]))

