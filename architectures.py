from keras.layers import Convolution2D, Dense, MaxPooling2D, Dropout, Input, Flatten, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop

def architecture_01(target_size):
    inputs = Input(shape=(512,512,1))

    x = ZeroPadding2D((1,1))(inputs)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(1024, kernel_initializer="glorot_uniform", activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, kernel_initializer="glorot_uniform", activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(target_size,activation="softmax")(x)

    model = Model(inputs = inputs, outputs=outputs)
    model.summary()

    optimizer = RMSprop(lr=0.001)
    
    if target_size == 2:
        loss = "binary_crossentropy"
    else:
        loss = "categorical_crossentropy"
    print(loss)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    return 1, model, "N{target_size:02d}".format(target_size=target_size)
    
def architecture_02():
    inputs = Input(shape=(512,512,1))

    x = ZeroPadding2D((1,1))(inputs)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256,(3,3), kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(1024, kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, kernel_initializer="glorot_uniform", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2,activation="softmax")(x)

    model = Model(inputs = inputs, outputs=outputs)
    model.summary()

    optimizer = Adam(lr=0.0001)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    return 2, model, ""