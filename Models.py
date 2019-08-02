
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.applications import ResNet50 as resnet


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
        input_shape = (self.height, self.width, self.channels)
        axis = -1
        # if using theano
        if K.image_data_format() == "channels_first":
            input_shape = (self.channels, self.height, self.width)
            axis = 1

        # conv_1
        _model.add(Conv2D(
            self.scale, (3, 3),
            padding="same",
            input_shape=input_shape)
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


class ResNet50(object):

    def __init__(self, height, width, channels, classes, parameter_scaling):
        self.height = height
        self.width = width
        self.channels = channels
        self.output_classes = classes
        self.scale = parameter_scaling

    def model(self, weights=None):
        _model = resnet(
            weights=weights,
            include_top=False,
            input_shape=(self.height, self.width, self.channels),
            pooling="max"
        )
        x = _model.output
        x = Flatten(input_shape=_model.output_shape[1:])(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(self.output_classes, activation="softmax")(x)
        model = Model(inputs=_model.input, outputs=x)
        return model
















