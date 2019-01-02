#  Visualizes the supplied 2-d training matrix.
from matplotlib import pyplot as plt
from showdigit import showdigit


def showdata(data, predicted, expected):
    # Plot the prediction
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    # fig.suptitle('KKK', fontsize=16)
    fig.subplots_adjust(left=0, right=1, bottom=0,
                        top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is 8x8 pixels
    for i in range(len(data)):
        ax = fig.add_subplot(32, 32, i + 1, xticks=[], yticks=[])
        ddd = showdigit(data[i], 32)
        if predicted[i] == expected[i]:
            ax.imshow(ddd, cmap=plt.cm.binary,
                      interpolation='nearest')
        else:
            ax.imshow(ddd, cmap=plt.cm.gist_heat,
                      interpolation='nearest')
