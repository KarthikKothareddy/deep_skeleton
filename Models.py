
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K


class CustomNet(object):

    def __init__(self, height, width, channels, classes, parameter_scaling):
        self.height = height
        self.width = width
        self.channels = channels
        self.output_classes = classes
        self.scale = parameter_scaling

    def model(self):
        # initiate model
        _model = Sequential()
        inputShape = (self.height, self.width, self.channels)
        axis = -1
        # if using theano
        if K.image_data_format() == "channels_first":
            input_shape = (self.channels, self.height, self.width)
            axis = 1

        # conv_1
        _model.add(Conv2D(
            self.scale, (3, 3),
            padding="same",
            input_shape=inputShape)
        )
        _model.add(Activation("relu"))
        _model.add(BatchNormalization(axis=axis))
        # conv_2
        _model.add(Conv2D(self.scale, (3, 3), padding="same"))
        _model.add(Activation("relu"))
        _model.add(BatchNormalization(axis=axis))
        # pool_1
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.25))

        # conv_3
        _model.add(Conv2D(self.scale*2, (3, 3), padding="same"))
        _model.add(Activation("relu"))
        _model.add(BatchNormalization(axis=axis))
        # conv_4
        _model.add(Conv2D(self.scale*2, (3, 3), padding="same"))
        _model.add(Activation("relu"))
        _model.add(BatchNormalization(axis=axis))
        # pool_2
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.25))

        # Fully connected layers
        _model.add(Flatten())
        _model.add(Dense(512))
        _model.add(Activation("relu"))
        _model.add(BatchNormalization())
        _model.add(Dropout(0.5))
        # classifier
        _model.add(Dense(self.output_classes))
        _model.add(Activation("softmax"))

        # return model
        return _model


















