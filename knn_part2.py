# PART 2 (FEATURE EXTRACTION)
import scipy.io as sio
import numpy as np
import sklearn.utils
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt
import time

from getonedigit import getonedigit
import extractfeatures
from knearest import knearest


def classify(k_values, test_data, train_data, test_labels, train_labels):
    accuracies = []
    for k in k_values:
        predicted = []
        for digit in test_data:
            predicted_label = knearest(
                k, digit, train_data, train_labels)
            predicted.append(predicted_label)

        # display confusion matrix for K = 5
        if k == 5:
            confusion_matrix(predicted, test_labels)

        # percentage of accurate predictions (accuracy)
        accuracy = sklearn.metrics.accuracy_score(
            test_labels, predicted)
        # add accuracy % to list of accuracies
        accuracies.append(accuracy)

    print(k_values)
    print(accuracies)

    return [predicted, accuracies]


def confusion_matrix(predicted, expected):
    '''
    calculates and displays a confusion matrix
    inputs: predicted labels, expected labels
    '''
    confusion_matrix = np.zeros((3, 3))
    for i in range(len(predicted)):
        if predicted[i] == 3 and expected[i] == 3:
            confusion_matrix[0][0] += 1
        elif predicted[i] == 6 and expected[i] == 3:
            confusion_matrix[0][1] += 1
        elif predicted[i] == 8 and expected[i] == 3:
            confusion_matrix[0][2] += 1
        if predicted[i] == 3 and expected[i] == 6:
            confusion_matrix[1][0] += 1
        elif predicted[i] == 6 and expected[i] == 6:
            confusion_matrix[1][1] += 1
        elif predicted[i] == 8 and expected[i] == 6:
            confusion_matrix[1][2] += 1
        if predicted[i] == 3 and expected[i] == 8:
            confusion_matrix[2][0] += 1
        elif predicted[i] == 6 and expected[i] == 8:
            confusion_matrix[2][1] += 1
        elif predicted[i] == 8 and expected[i] == 8:
            confusion_matrix[2][2] += 1

    sum_rows = np.sum(confusion_matrix, axis=1).tolist()

    print('CONFUSION MATRIX')
    print('  \t3\t6     8')
    n = ['3', '6', '8']
    i = 0
    for row in confusion_matrix:
        print(n[i] + '\t' + str(row) + ' Accuracy of ' + n[i] +
              ' = ' + str(confusion_matrix[i][i] / sum_rows[i]))
        i += 1


def plot_accuracies(ks, accuracies, legends):
    sns.set(style="darkgrid", rc={'axes.facecolor': '#EAEAF4',
                                  'figure.facecolor': '#EAEAF4'})
    plt.figure()

    line_options = ['bs-', 'gs-', 'rs-', 'cs-', 'ms-', 'ks-', 'ys-']
    for i in range(0, len(accuracies)):
        plt.plot(ks, accuracies[i], line_options[i],
                 label=legends[i])

    plt.legend(loc='best')

    plt.xlabel('K', fontweight='bold')
    plt.ylabel('Testing Accuracy', fontweight='bold')
    plt.xticks(k_values, ('1', '3', '5', '7',
                          '9', '11', '13', '15', '17', '19'))
    plt.title('Comparison of Accuracy for different features (K varies from 1 to 20)',
              fontdict={'fontsize': 12, 'fontweight': 'bold'})

    plt.savefig('feature_accuracies_comparison.png',
                bbox_inches='tight')
    plt.show()


# load all main and benchmark datasets
mainData = sio.loadmat('usps_main.mat')
benchMarkData = sio.loadmat('usps_benchmark.mat')
training_set = mainData['maindata']
test_set = benchMarkData['benchmarkdata']

# TRAINING DATA :-
#  get 500 digits for 3,6,8 each from usps_main.mat ( training set )
# TESTING DATA :-
# get 500 digits for 3,6,8 each from usps_benchmark.mat ( testing set )
train_set_3 = []
train_set_6 = []
train_set_8 = []
test_set_3 = []
test_set_6 = []
test_set_8 = []
for i in range(0, 500):
    train_set_3.append(np.reshape(
        getonedigit(2, i, training_set), (256, 1)))
    train_set_6.append(np.reshape(
        getonedigit(5, i, training_set), (256, 1)))
    train_set_8.append(np.reshape(
        getonedigit(7, i, training_set), (256, 1)))
    test_set_3.append(np.reshape(
        getonedigit(2, i, test_set), (256, 1)))
    test_set_6.append(np.reshape(
        getonedigit(5, i, test_set), (256, 1)))
    test_set_8.append(np.reshape(
        getonedigit(7, i, test_set), (256, 1)))

# arrays to store accuracies for different feature vectors
f_accuracies = []
f_legends = []

'''
256 features
'''
# choose 1500 samples 3 , 6,  8  for training
train_data = train_set_3 + train_set_6 + train_set_8
# create training labels
train_labels = [3] * 500 + [6] * 500 + [8] * 500
# shuffle rows (both data and corresponding labels data)
train_data, train_labels = sklearn.utils.shuffle(
    train_data, train_labels)

# choose 1500 samples  3 , 6,  8  for testing
test_data = test_set_3 + test_set_6 + test_set_8
# create test labels (expected values)
test_labels = [3] * 500 + [6] * 500 + [8] * 500
# shuffle rows (both data and corresponding labels data)
test_data, test_labels = sklearn.utils.shuffle(test_data, test_labels)

# initialize k values from 1 to 20
max_k = 20
k_values = [k for k in range(1, max_k, 2)]


'''
256 features (NORMALIZED)
'''
normalized_test_data = []
normalized_train_data = []
for i in range(0, len(test_data)):
    normalized_test_data.append(
        extractfeatures.normalize(test_data[i]))
    normalized_train_data.append(
        extractfeatures.normalize(train_data[i]))


# apply KNN rule(varying K from 1 to 20)
t1 = time.clock()
predicted, accuracies = classify(
    k_values, normalized_test_data, normalized_train_data, test_labels, train_labels)
t2 = time.clock()
print('TIME TAKEN 256 = ' + str(t2 - t1))

f_accuracies.append(accuracies)
f_legends.append('Normalized Features: 256')

'''
Principal Component Analysis applied to 256 size vectors
'''
# variances contains the percent of variance to be retained from origina
variances = [0.90, 0.70]
for variance_retained in variances:
    pca_test_data, pca_train_data = extractfeatures.get_pca(
        test_data, train_data, variance_retained)

    # apply KNN rule(varying K from 1 to 20)
    t1 = time.clock()
    predicted, accuracies = classify(
        k_values, pca_test_data, pca_train_data, test_labels, train_labels)
    t2 = time.clock()
    print('TIME TAKEN PCA = ' + str(t2 - t1) + '  ' +
          str(len(pca_test_data[0])))

    f_accuracies.append(accuracies)
    f_legends.append(str(variance_retained * 100) + '%' + ' PCA ' +
                     'Features: ' + str(len(pca_test_data[0])))


'''
16 features (NORMALIZED)
'''
normalized_test_data16 = []
normalized_train_data16 = []
for i in range(0, len(test_data)):
    test_16 = extractfeatures.extract16(
        np.reshape(test_data[i], (16, 16)))
    normalized_test_data16.append(extractfeatures.normalize(test_16))
    train_16 = extractfeatures.extract16(
        np.reshape(train_data[i], (16, 16)))
    normalized_train_data16.append(
        extractfeatures.normalize(train_16))


# apply KNN rule(varying K from 1 to 20)
predicted, accuracies = classify(
    k_values, normalized_test_data16, normalized_train_data16, test_labels, train_labels)

f_accuracies.append(accuracies)
f_legends.append('Normalized Features: 16')

# plot the accuracies obtained from different feature sets
plot_accuracies(k_values, f_accuracies, f_legends)
