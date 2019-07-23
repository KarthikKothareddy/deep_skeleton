
import os
import numpy as np
import cv2


class DataSetLoader(object):

    def __init__(self, input_path):
        self.input_path = input_path

    def get_data(self):
        # input_path = self.input_path if input_path is None else input_path
        # input and target labels
        X = []
        y = []
        for i, path in enumerate(self.input_path):
            # valid given the input -> /path/to/data/{class}/{image}.jpg
            _X = cv2.imread(path)
            _y = path.split(os.path.sep)[-2]
            X.append(_X)
            y.append(_y)
        return np.array(X), np.array(y)
