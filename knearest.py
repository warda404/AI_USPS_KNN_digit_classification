'''
K-nearest Neighbour classifier
Y = KNEAREST( k, x, data, truelabels )

Arguments:
'data' should be a N rows by M columns matrix of data, composed
of N training examples, each with M dimensions.

'truelabels' should be a Nx1 column vector, with class labels.

'x' is the TEST data vector, size 1xM, where the knn estimate is required.

'k' is the number of neighbours to take into account.
Note that even values will result in ties broken randomly.

Returns:
'y' - a predicted class label for your data vector 'x'
'''

import numpy as np
import scipy
import random
from collections import Counter


def knearest(k, x, data, datalabels):
    numtrain = len(data)  # rows (number of training examples)
    # columns (dimension size of each example)
    numfeatures = len(data[0])

    if len(x) != numfeatures:
        exit('Test data dimensions does not match train data dimensions.')

    if k > numtrain:
        exit('Not enough training samples to use k = ' +
             str(k) + ' (you only supplied ' + str(numtrain) + ')')

    # measure Euclidean distance from this test example
    # to every training example
    distances = []
    for i in range(numtrain):
        # first we compute the euclidean distance
        distance = np.sqrt(abs(np.sum(np.square(x - data[i]))))
        # add it to list of distances
        distances.append([distance, datalabels[i]])

    distances = sorted(distances)

    # get k closest values
    closest = []
    for i in range(0, k):
        closest.append(distances[i][1])

    # check if tie exists
    # unique_classes = list(Counter(closest).keys())
    # class_counts = list(Counter(closest).values())
    # tie_exists = all(x == class_counts[0] for x in class_counts)
    #
    # if tie_exists:
    #     y = random.choice(unique_classes)
    #     return y
    # else:
    #     y = (scipy.stats.mode(closest, axis=0))
    #     return y[0]

    y = (scipy.stats.mode(closest, axis=0))
    return y[0]
