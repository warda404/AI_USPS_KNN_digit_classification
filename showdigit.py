import cv2
import numpy as np


def get_square(image, square_size):

    height, width = image.shape
    if(height > width):
        differ = height
    else:
        differ = width
    differ += 4

    mask = np.zeros((differ, differ), dtype="uint8")
    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)
    mask[y_pos:y_pos + height, x_pos:x_pos +
         width] = image[0:height, 0:width]
    mask = cv2.resize(mask, (square_size, square_size),
                      interpolation=cv2.INTER_AREA)

    return mask


def showdigit(digit, factor):
    '''
    Displays digit as image
    Digit must be 16*16 or 256*1 array
    '''
    digit = np.reshape(digit, (16, 16))
    magnified_digit = get_square(digit, factor)
    # cv2.imshow('digit', magnified_digit)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return magnified_digit
