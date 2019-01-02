import numpy as np
import sklearn.decomposition
from sklearn.preprocessing import StandardScaler


def extract16(digdata):
    # sum the values  along matrix dimension 1 (rows)
    feature_vector = digdata.sum(axis=0)
    return feature_vector


def normalize(digdata):
    feature_vector = []
    for i in range(0, len(digdata)):
        feature_vector.append(
            (digdata[i] - np.mean(digdata)) / np.std(digdata))

    feature_vector = np.array(feature_vector)
    feature_vector = np.reshape(feature_vector, (len(digdata), 1))
    return feature_vector


def get_pca(test_data, train_data, variance_retained):

    train_data = np.reshape(train_data, (1500, 256))
    test_data = np.reshape(test_data, (1500, 256))

    # standardize data first (normalize)
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(train_data)
    # Apply transform to both the training set and the test set.
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # choose the minimum number of principal components such that x% of the variance is retained
    pca_model = sklearn.decomposition.PCA(variance_retained)
    pca_model.fit(train_data)
    print(pca_model.n_components_)

    train_data = pca_model.transform(train_data)
    test_data = pca_model.transform(test_data)

    return [test_data, train_data]
