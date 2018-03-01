import os
import numpy as np
import pandas as pd
from keras import Input
from keras.applications import ResNet50, xception
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.engine import Model
from keras.models import Sequential
from keras.optimizers import Adam

import src.util as util
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, Dropout

CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
NUM_CATEGORIES = len(CATEGORIES)

SAMPLE_PER_CATEGORY = 200
SEED = 1987
data_dir = '../../res/'
train_dir = os.path.join(data_dir, 'train_npy')
test_dir = os.path.join(data_dir, 'test')

np.random.seed(seed=SEED)
for category in CATEGORIES:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))

train = []
val = []
for category_id, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(train_dir, category)):
        if (np.random.random(1) < 0.9)[0]:
            train.append(['{}/{}'.format(category, file), category_id, category])
        else:
            val.append(['{}/{}'.format(category, file), category_id, category])

input_layer = Input(shape=(484, 484, 3))
xception_model = xception.Xception(include_top=False, weights='imagenet', input_tensor=input_layer, input_shape=(484, 484, 3), pooling='max')
print(len(xception_model.layers))
for layer in xception_model.layers[:50]:
   layer.trainable = False
for layer in xception_model.layers[50:]:
   layer.trainable = True

for lay in xception_model.layers:
    layer.trainable = True

#curr_layer = resnet()(input_layer)
print(xception_model.output)
#flt = Flatten()(xception_model.output)
dense1 = Dense(512, activation='relu', use_bias=True)(xception_model.output)
# dropout1 = Dropout(.2)(dense1)
# print(dense1._keras_shape, "dense1")
# dense2 = Dense(256, activation='relu', use_bias=True)(dropout1)
# print(dense2._keras_shape, "dense2")
# dense3 = Dense(64, activation='relu', use_bias=True)(dense2)
# print(dense3._keras_shape, "dense3")
predictions = Dense(12, activation='softmax')(xception_model.output)
print(predictions._keras_shape, "predictions")

learning_rate = 0.001
decay_rate = learning_rate / 32
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)


model = Model(inputs=input_layer, outputs=predictions)
#model.load_weights("../models/tmp/res50_trans_net_test_check_fulltrain.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


checkpointer = ModelCheckpoint(filepath="keras_xception_check_2.hdf5", verbose=1, save_best_only=True)
model.fit_generator(util.load_batch_keras(train_dir, train, batch_size=8, rotate_image=True),
                    steps_per_epoch=len(train) // 32,
                    validation_data=util.load_batch_keras(train_dir, val, batch_size=8),
                    validation_steps=len(val) // 32, epochs=60, verbose=1, callbacks=[checkpointer])


model.save('keras_trans_5.h5')

