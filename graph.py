#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import division, print_function
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.layers import Input, Conv1D, BatchNormalization, Activation, Dropout, MaxPooling1D, Lambda, add, Dense, TimeDistributed
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, Dense, add, GlobalAveragePooling1D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from config import get_config


# In[8]:


def ECG_model(config):
    inputs = Input(shape=(config.input_size, 1), name='input')

    # Initial Conv layer
    x = Conv1D(filters=config.filter_length, kernel_size=config.kernel_size, padding='same', strides=1, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Residual Blocks
    for i in range(30):  # Increase the number of layers
        shortcut = x
        if i % 3 == 0:
            x = Conv1D(filters=config.filter_length, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal')(x)
        else:
            x = Conv1D(filters=config.filter_length, kernel_size=config.kernel_size, padding='same', strides=1, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(config.drop_rate)(x)
        x = Conv1D(filters=config.filter_length, kernel_size=config.kernel_size, padding='same', strides=1, kernel_initializer='he_normal')(x)
        
        if i % 3 == 0:
            shortcut = Conv1D(filters=config.filter_length, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal')(shortcut)
        x = add([x, shortcut])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)

    # Final Dense layer
    classes = ['N', 'V', '/','R','A','L','S']
    len_classes = len(classes)
    x = Dense(len_classes, activation='softmax')(x)
    outputs = Reshape((1, len_classes))(x)  # Reshape to (batch_size, 1, 4)

    model = Model(inputs=inputs, outputs=outputs)
    adam = Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# In[ ]:




