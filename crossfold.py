from sklearn.model_selection import train_test_split


def split_data(train_data, labels, test_size):
    all_data = []
    for i in range(0, len(train_data)):
        all_data.append((train_data[i], labels[i]))

    trainn, testt = train_test_split(all_data, test_size=test_size)
    train = []
    train_labels = []
    test = []
    test_labels = []
    for i in range(0, len(trainn)):
        train.append(trainn[i][0])
        train_labels.append(trainn[i][1])
        test.append(testt[i][0])
        test_labels.append(testt[i][1])

    return [test, train, test_labels, train_labels]
