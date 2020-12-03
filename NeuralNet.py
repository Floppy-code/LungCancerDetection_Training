from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten, Concatenate
from keras.callbacks import EarlyStopping

NN_SHAPE = (256, 256, 1)

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

    model.add(Dense(256))
    model.add(Activation('softmax'))

    model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model


def get_neural_net_WH():
    """NN used for true/false identification of nodules"""
    
    #PROOF OF CONCEPT MODEL
    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape = NN_SHAPE, padding = 'same')) #Width, Height, Colors
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(128, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (4,4)))

    model.add(Conv2D(256, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dense(64 + 1))
    model.add(Activation('softmax'))

    model.compile(optimizer = 'Adamax', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model