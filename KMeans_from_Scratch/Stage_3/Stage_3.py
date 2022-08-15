import numpy as np
from sklearn.datasets import load_wine
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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

    kmeans = CustomKMeans(2)
    kmeans.fit(X_full)
    predicted_labels = kmeans.predict(X_full)
    print(list(predicted_labels[:10]))
