import scipy.io as sio
import numpy as np
import random
import sklearn.utils
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt

from getonedigit import getonedigit
from knearest import knearest
from crossfold import split_data
from showdata import showdata


def predict_and_get_accuracy(k_values, test_data, train_data, test_labels, train_labels):
    '''
    returns accuracies corresponding to each value of K
    '''
    accuracies = []
    for k in k_values:
        predicted = []
        for digit in test_data:
            predicted_label = knearest(
                k, digit, train_data, train_labels)
            predicted.append(predicted_label)

        # percentage of accurate predictions (accuracy)
        accuracy = sklearn.metrics.accuracy_score(
            test_labels, predicted)
        # add accuracy % to list of accuracies
        accuracies.append(accuracy)

    print(k_values)
    print(accuracies)
    return accuracies

# functions to plot accuracies


def plot_accuracy(ks, accuracies, choice, img_path):
    sns.set(style="darkgrid", rc={'axes.facecolor': '#EAEAF4',
                                  'figure.facecolor': '#EAEAF4'})
    plt.figure()
    plt.plot(ks, accuracies, 'bs-')

    plt.xlabel('K', fontweight='bold')
    plt.ylabel(choice + ' Accuracy', fontweight='bold')
    plt.xticks(ks, ('1', '3', '5', '7',
                    '9', '11', '13', '15', '17', '19'))
    plt.title(choice + ' Accuracy (K varies from 1 to 20)',
              fontdict={'fontsize': 12, 'fontweight': 'bold'})

    plt.savefig(img_path, bbox_inches='tight')
    plt.show()


def plot_accuracies(ks, testing_accuracies_1, testing_accuracies_2):
    sns.set(style="darkgrid", rc={'axes.facecolor': '#EAEAF4',
                                  'figure.facecolor': '#EAEAF4'})
    plt.figure()
    plt.plot(ks, testing_accuracies_1, 'bs-',
             label='testing accuracy 1')
    plt.plot(ks, testing_accuracies_2, 'gs-',
             label='testing accuracy 2')
    plt.legend(loc='best')

    plt.xlabel('K', fontweight='bold')
    plt.ylabel('Testing Accuracies', fontweight='bold')
    plt.xticks(ks, ('1', '3', '5', '7',
                    '9', '11', '13', '15', '17', '19'))
    plt.title('Comparison of Testing Accuracies (K varies from 1 to 20)',
              fontdict={'fontsize': 12, 'fontweight': 'bold'})

    plt.savefig('testing_accuracies_comparison.png',
                bbox_inches='tight')
    plt.show()


# load maindata dataset
loaded_data = sio.loadmat('usps_main.mat')
main_data = loaded_data['maindata']

# get 500 digits for 3 and 8 respectively from main dataset
main_set_3 = []
main_set_8 = []
for i in range(0, 500):
    main_set_3.append(np.reshape(
        getonedigit(2, i, main_data), (256, 1)))
    main_set_8.append(np.reshape(
        getonedigit(7, i, main_data), (256, 1)))

# choose 100 random unique samples of 3 and 8  for training
train_3 = random.sample(main_set_3, 100)
train_8 = random.sample(main_set_8, 100)
train_data = train_3 + train_8
# create label vector
labels = [3] * 100 + [8] * 100

# shuffle rows (both data and corresponding labels data)
train_data, labels = sklearn.utils.shuffle(train_data, labels)

# apply KNN rule for each digit in training data (varying K from 1 to 20)
max_k = 20
k_values = [k for k in range(1, max_k, 2)]
training_accuracies = predict_and_get_accuracy(
    k_values, train_data, train_data, labels, labels)

# Plot training accuracy vs K
plot_accuracy(k_values, training_accuracies,
              'Training', 'training_accuracy.png')

# ******************************************************************************
# split data into 2 random equal parts
test, train, test_labels, train_labels = split_data(
    train_data, labels, 0.5)

# apply KNN rule for test data (varying K from 1 to 20)
testing_accuracies_1 = predict_and_get_accuracy(
    k_values, test, train, test_labels, train_labels)

# Plot testing accuracy vs K
plot_accuracy(k_values, testing_accuracies_1,
              'Testing', 'testing_accuracy_1.png')

# split data into 2 random equal parts again
test, train, test_labels, train_labels = split_data(
    train_data, labels, 0.5)

# apply KNN rule for test data (varying K from 1 to 20)
testing_accuracies_2 = predict_and_get_accuracy(
    k_values, test, train, test_labels, train_labels)

# Plot testing  accuracy vs K
plot_accuracy(k_values, testing_accuracies_2,
              'Testing', 'testing_accuracy_2.png')

# compare the two testing accuracies from two random splits
plot_accuracies(k_values, testing_accuracies_1, testing_accuracies_2)

# plot average accuracy and standard deviation as error bars
avg_accuracies = []
standard_deviations = []
for i in range(0, len(testing_accuracies_1)):
    a = testing_accuracies_1[i]
    b = testing_accuracies_2[i]
    avg_accuracy = (a + b) / 2
    avg_accuracies.append(avg_accuracy)
    std = abs(a - b) / 2
    standard_deviations.append(std)


# plt.errorbar(avg_accuracies, k_values,
#              xerr=avg_accuracies)
# # plt.ylabel('standard deviation')
# # plt.xlabel('K')
# plt.title('error bar')
# plt.show()

# extending above to load and predict 3, 6, 8
# get 500 digits for 3,6,8 each from usps_main.mat
main_set_3 = []
main_set_6 = []
main_set_8 = []
for i in range(0, 500):
    # all feature vectors will be 256*1
    main_set_3.append(np.reshape(
        getonedigit(2, i, main_data), (256, 1)))
    main_set_6.append(np.reshape(
        getonedigit(5, i, main_data), (256, 1)))
    main_set_8.append(np.reshape(
        getonedigit(7, i, main_data), (256, 1)))


# choose 100 random unique samples of 3, 6, 8 from total data for training
train_3 = random.sample(main_set_3, 100)
train_6 = random.sample(main_set_6, 100)
train_8 = random.sample(main_set_8, 100)
train_data = train_3 + train_6 + train_8
# create label vector
labels = [3] * 100 + [6] * 100 + [8] * 100

# shuffle rows (both data and corresponding labels data)
train_data, labels = sklearn.utils.shuffle(train_data, labels)

# apply KNN rule for each digit in training data (varying K from 1 to 20)
training_accuracies = predict_and_get_accuracy(
    k_values, train_data, train_data, labels, labels)

# Plot training accuracy vs K
plot_accuracy(k_values, training_accuracies,
              'Training', 'training_accuracy_6.png')

# ******************************************************************************
# split data into 2 random equal parts
test, train, test_labels, train_labels = split_data(
    train_data, labels, 0.5)

# apply KNN rule for test data (varying K from 1 to 20)
testing_accuracies_1 = predict_and_get_accuracy(
    k_values, test, train, test_labels, train_labels)

# Plot testing accuracy vs K
plot_accuracy(k_values, testing_accuracies_1,
              'Testing', 'testing_accuracy_1_6.png')

# split data into 2 random equal parts again
test, train, test_labels, train_labels = split_data(
    train_data, labels, 0.5)

# apply KNN rule for test data (varying K from 1 to 20)
testing_accuracies_2 = predict_and_get_accuracy(
    k_values, test, train, test_labels, train_labels)

# Plot testing  accuracy vs K
plot_accuracy(k_values, testing_accuracies_2,
              'Testing', 'testing_accuracy_2_6.png')

# compare the two testing accuracies from two random splits
plot_accuracies(k_values, testing_accuracies_1, testing_accuracies_2)

# plot average accuracy and standard deviation as error bars
avg_accuracies = []
standard_deviations = []
for i in range(0, len(testing_accuracies_1)):
    a = testing_accuracies_1[i]
    b = testing_accuracies_2[i]
    avg_accuracy = (a + b) / 2
    avg_accuracies.append(avg_accuracy)
    std = abs(a - b) / 2
    standard_deviations.append(std)

width = 0.38
plt.bar(k_values, avg_accuracies, width, yerr=standard_deviations)
plt.ylabel('Average accuracy')
plt.xlabel('K')
plt.title('Average accuracy (standard deviation as error bar)')
plt.xticks(k_values, ('1', '3', '5', '7',
                      '9', '11', '13', '15', '17', '19'))
plt.savefig('average_accuracies_3_6_8.png', bbox_inches='tight')
plt.show()
