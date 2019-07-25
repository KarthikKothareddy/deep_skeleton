
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
from imutils import paths
import imutils


class DataSetLoader(object):
    """
    This class contains utilities to loads a dataset from a specified
    path given that the inputs are in the path of below form
        /path/to/data/{target_class_1}/{image_1}.jpg ... {image_n}.jpg
        /path/to/data/{target_class_2}/{image_1}.jpg ... {image_n}.jpg
        ...
        ...
        /path/to/data/{target_class_n}/{image_1}.jpg ... {image_n}.jpg

    """
    def __init__(self, input_path, preprocessors=None):
        self.input_path = input_path
        self.preprocessors = preprocessors

    def get_data(self, **kwargs):
        """
        Returns the data in the form of (images, labels) by parsing
        an input directory
        :param kwargs:
        :return: Two numpy arrays containing images and target labels
        """
        # input and target labels
        X = []
        y = []
        images = list(imutils.paths.list_images(self.input_path)) \
            if kwargs["expand_path"] \
            else self.input_path
        for i, path in enumerate(images):
            _X = cv2.imread(path)
            _y = path.split(os.path.sep)[-2]
            # if any preprocessors
            if self.preprocessors:
                for p in self.preprocessors:
                    _X = p._preprocess(_X)
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

    def _preprocess(self, image):
        return img_to_array(image, data_format=self.data_format)


class RescalePreprocessor(object):
    """
    This class scales image to the given width and height, also
    one can choose to keep the aspect ratio intact after resizing
    """
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def _preprocess(self, image, **kwargs):
        """
        Resizes the given image according to new dimensions, one can
        choose to keep or ignore aspect ratio of original image by
        passing the aspect_aware flag

        :param image: The input image
        :param kwargs:
        :return: resized image according to new dimensions
        """
        # if interpolation is specified then use it or else use INTER_AREA
        interpolation = kwargs["interpolation"] if kwargs["interpolation"] \
            else cv2.INTER_AREA
        # if specified to preserve aspect ratio
        if kwargs["aspect_aware"]:
            h, w = image.shape[:2]
            dW = 0
            dH = 0
            # if width is smaller than height
            if w < h:
                image = imutils.resize(
                    image,
                    width=self.width,
                    inter=interpolation
                )
                dH = int((image.shape[0] - self.height) / 2.0)
            # if height is smaller than width
            else:
                image = imutils.resize(
                    image,
                    height=self.height,
                    inter=interpolation
                )
                dW = int((image.shape[1] - self.width) / 2.0)
            h, w = image.shape[:2]
            image = image[dH:h - dH, dW:w - dW]
        # finally return the resized image
        return cv2.resize(
            image,
            (self.width, self.height),
            interpolation=interpolation
        )


def _bgr_to_rgb(self, images):
    return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

def _rgb_to_gray(self, images):
    return [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images]

def _rgb_to_hsv(self, images):
    return [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in images]






