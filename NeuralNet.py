from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping

NN_SHAPE = (512, 512, 1)

def get_neural_net_WH():
    """NN used for true/false identification of nodules"""
    
    #PROOF OF CONCEPT MODEL
    model = Sequential()
    model.add(Conv2D(16, (5,5), input_shape = NN_SHAPE, padding = 'same')) #Width, Height, Colors
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5,5), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(16, (5,5), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5,5), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(16, (5,5), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5,5), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model