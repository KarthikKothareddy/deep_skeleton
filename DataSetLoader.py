
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from imutils import paths


class DataSetLoader(object):

    def __init__(self, input_path):
        self.input_path = input_path

    def get_data(self, **kwargs):
        # input_path = self.input_path if input_path is None else input_path
        # input and target labels
        X = []
        y = []
        images = list(paths.list_images(self.input_path)) \
            if kwargs["expand_path"] \
            else self.input_path
        for i, path in enumerate(images):
            # valid given the input ->
            # /path/to/data/
            #   {target_class_1}/{image}.jpg
            #   {target_class_2}/{image}.jpg
            _X = cv2.imread(path)
            _y = path.split(os.path.sep)[-2]
            X.append(_X)
            y.append(_y)
        if kwargs["encode_labels"]:
            encoder = LabelEncoder()
            y = encoder.fit_transform(np.array(y))
        return np.array(X), y

