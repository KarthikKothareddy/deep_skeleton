
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
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
        :param kwargs: extend_path and encode_labels are optional
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
                print("INFO: Pre-processing inputs...")
                for p in self.preprocessors:
                    _X = p.run(_X)
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

    def _image_to_array(self, image):
        return img_to_array(image, data_format=self.data_format)

    def run(self, image):
        return self._image_to_array(image)


class ChangeColorSpace(object):
    """
    This class contains utilities to convert image from one color
    space to another assuming the source to target conversion is
    valid in cv2
    """
    cmaps = {
        "BGR": {
            "RGB": cv2.COLOR_BGR2RGB,
            "GRAY": cv2.COLOR_BGR2GRAY,
            "HSV": cv2.COLOR_BGR2HSV,
            "LAB": cv2.COLOR_BGR2LAB
        },
        "RGB": {
            "BGR": cv2.COLOR_RGB2BGR,
            "GRAY": cv2.COLOR_RGB2GRAY,
            "HSV": cv2.COLOR_RGB2HSV,
            "LAB": cv2.COLOR_RGB2LAB
        },
        "HSV": {
            "BGR": cv2.COLOR_HSV2BGR,
            "RGB": cv2.COLOR_HSV2RGB
        }
    }

    def __init__(self, source, target):
        self.source = str(source).upper()
        self.target = str(target).upper()

    def _convert_space(self, image):
        """
        Changes the color space of the image to a specified space
        :param image: The input image
        :return: Modified image with new color space
        """
        try:
            # source to target
            return cv2.cvtColor(image, self.cmaps[self.source][self.target])
        except KeyError:
            print(
                "ERROR: The specified color space conversion does not exist..."
            )
            return None

    def run(self, image):
        return self._convert_space(image)


class Rescale(object):
    """
    This class scales image to the given width and height, also
    one can choose to keep the aspect ratio intact after resizing
    """
    def __init__(self, height, width, **kwargs):
        self.height = height
        self.width = width
        # if interpolation is specified then use it or else use INTER_AREA
        self.interpolation = kwargs.get("interpolation", cv2.INTER_AREA)
        # aspect_aware is turned off by default
        self.aspect_aware = kwargs.get("aspect_aware", False)

    def _change_scale(self, image):
        """
        Resizes the given image according to new dimensions, one can
        choose to keep or ignore aspect ratio of original image by
        passing the aspect_aware flag in constructor

        :param image: The input image
        :return: resized image according to new dimensions
        """
        # if specified to preserve aspect ratio
        if self.aspect_aware:
            h, w = image.shape[:2]
            dW = 0
            dH = 0
            # if width is smaller than height
            if w < h:
                image = imutils.resize(
                    image,
                    width=self.width,
                    inter=self.interpolation
                )
                dH = int((image.shape[0] - self.height) / 2.0)
            # if height is smaller than width
            else:
                image = imutils.resize(
                    image,
                    height=self.height,
                    inter=self.interpolation
                )
                dW = int((image.shape[1] - self.width) / 2.0)
            h, w = image.shape[:2]
            image = image[dH:h - dH, dW:w - dW]
        # finally return the resized image
        return cv2.resize(
            image,
            (self.width, self.height),
            interpolation=self.interpolation
        )

    def run(self, image):
        return self._change_scale(image)


class GaussianBlur(object):
    """
    Applies Gaussian blur to an input image and returns the resulting
    image. One can choose the kernel size and sigmaX for the filter
    """
    def __init__(self, kernel, sigmaX):
        self.kernel = kernel
        self.sigmaX = sigmaX

    def _apply_blur(self, image):

        image = cv2.GaussianBlur(
            image, ksize=self.kernel, sigmaX=self.sigmaX
        )
        return image

    def run(self, image):
        return self._apply_blur(image)


class HistogramEqualize(object):

    def __init__(self, **kwargs):
        self.channel_wise = kwargs.get("channel_wise", False)

    def _histogram_equalize(self, image):
        return cv2.equalizeHist(image)

    def run(self, image):
        return self._histogram_equalize(image)


