from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten, Concatenate
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

NN_SHAPE = (128, 128, 1)

def get_neural_net_VGG16():
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,1), filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(64,(3,3),padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(256, (3,3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3,3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(4096,activation="relu"))
    model.add(Dense(4096,activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer = Adam(lr = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model

def get_neural_net_VGG19():
    #VGG-19 edited implementation
    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape = NN_SHAPE, padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), input_shape = NN_SHAPE, padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(128, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(256, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(512, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(512, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model


def get_neural_net_WH():
    """NN used for true/false identification of nodules"""
    
    #PROOF OF CONCEPT MODEL
    model = Sequential()
    
    #256px in
    model.add(Conv2D(8, (3,3), input_shape = NN_SHAPE, padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (4,4)))
    model.add(Dropout(0.2))

    model.add(Conv2D(16, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (4,4)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer = Adam(lr = 0.00001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model