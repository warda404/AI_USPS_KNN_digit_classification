import numpy as np
import pandas as pd


def getonedigit(digit, instance, training_data):
    '''
    instance : which instance of digit data from 500 samples (0 to 499)
    digit    : digit from 0 to 9 (0 -> 1, 1 -> 2, 2 -> 3 , ... , 9 -> 0)

    '''
    digit_256_1 = pd.DataFrame(
        np.hstack((training_data[:, instance, digit])))
    # print(digit_256_1.shape)
    digit_256_1 = np.array(digit_256_1)
    digit_16_16 = np.reshape(digit_256_1, (16, 16))
    digit_16_16 = np.flipud(np.rot90(digit_16_16))
    return digit_16_16

# digit = getonedigit(9, 311, training_data)
# print(digit)
