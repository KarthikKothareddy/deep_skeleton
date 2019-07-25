
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
from imutils import paths


class DataSetLoader(object):

    def __init__(self, input_path, preprocessors=None):
        self.input_path = input_path
        self.preprocessors = preprocessors

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
            # if any preprocessors
            if self.preprocessors:
                for p in self.preprocessors:
                    _X = p.preprocess(_X)
            X.append(_X)
            y.append(_y)
        y = np.array(y)
        if kwargs["encode_labels"]:
            print("INFO: Encoding labels...")
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        return np.array(X), y


class BasePreprocessor(object):
    """
    This is a base preprocessor which converts an image
    to correct format by using keras's img_to_array
    """
    def __init__(self, data_format="channels_last"):
        self.data_format = data_format

    def preprocess(self, image):
        return img_to_array(image, data_format=self.data_format)


def _bgr_to_rgb(self, images):
    return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

def _rgb_to_gray(self, images):
    return [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images]

def _rgb_to_hsv(self, images):
    return [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in images]






