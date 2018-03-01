import os
import numpy as np
import pandas as pd
import src.util as ut
import tensorflow as tf

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
        if (np.random.random(1)<0.8)[0]:
            train.append(['{}/{}'.format(category, file), category_id, category])
        else:
            val.append(['{}/{}'.format(category, file), category_id, category])


ut.load_batch(train_dir, train)
