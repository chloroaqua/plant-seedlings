import random

import scipy
from keras.preprocessing import image
import numpy as np
import os
import cv2
from scipy.ndimage import rotate

def load_batch(base_dir, train_labels, batch_size=32, dim=(484,484,3), category_size=12):
    batch_labels = np.zeros((batch_size, category_size))
    batch_data = np.zeros((batch_size, *dim))
    print(batch_data.shape)
    random_idx = np.random.choice(len(train_labels), batch_size, replace=True)
    for i, train_idx in enumerate(random_idx):
        print(i, train_idx)
        y = train_labels[train_idx][1]
        image_path = os.path.join(base_dir, train_labels[train_idx][0])
        image_data = np.load(image_path)
        batch_data[i,:] = image_data
        y_one_hot = np.zeros((category_size))
        y_one_hot[y] = 1
        batch_labels[i] = y_one_hot
    return batch_data, batch_labels


def load_batch_keras(base_dir, train_labels, batch_size=32, dim=(484,484,3), category_size=12, rotate_image=False):
    while True:
        batch_labels = np.zeros((batch_size, category_size))
        batch_data = np.zeros((batch_size, *dim))
        #print(batch_data.shape)
        random_idx = np.random.choice(len(train_labels), batch_size, replace=True)
        for i, train_idx in enumerate(random_idx):
            #print(i, train_idx)
            y = train_labels[train_idx][1]
            image_path = os.path.join(base_dir, train_labels[train_idx][0])
            image_data = np.load(image_path)
            if rotate_image:
                image_data = scipy.ndimage.interpolation.rotate(image_data, random.uniform(0, 360), reshape=False)
            image_data = ((image_data - (255.0 / 2)) / 255.0)
            batch_data[i, :] = image_data
            y_one_hot = np.zeros((category_size))
            y_one_hot[y] = 1
            batch_labels[i] = y_one_hot
        yield batch_data, batch_labels


def save_numpy(base_dir, output_dir, CATEGORIES, dim=(484,484)):
    for category_id, category in enumerate(CATEGORIES):
        for file in os.listdir(os.path.join(base_dir, category)):
            current_out_filename = os.path.join(output_dir, '{}/{}.npy'.format(category, file))
            current_image = os.path.join(base_dir, '{}/{}'.format(category, file))
            if not os.path.exists(os.path.join(output_dir, '{}'.format(category))):
                os.makedirs(os.path.join(output_dir, '{}'.format(category)))
            print(current_out_filename)
            cv_image = cv2.imread(current_image)
            cv_image = cv2.resize(cv_image, dim)
            np.save(current_out_filename, cv_image)


def read_img(filepath, size, data_dir):
    img = image.load_img(np.os.path.join(data_dir, filepath), target_size=size)
    img = image.img_to_array(img)
    return img


if __name__ == "__main__":
    CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                  'Loose Silky-bent',
                  'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
    print(len(CATEGORIES))
    #save_numpy('../res/train', '../res/train_npy', CATEGORIES)
