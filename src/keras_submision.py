import keras
import numpy as np
import os
import cv2
import csv

model = keras.models.load_model('keras_trans_5.h5')
model.load_weights("keras_xception_check_2.hdf5")
image_data = np.load('../../res/train_npy/Black-grass/0ace21089.png.npy')
image_data = ((image_data - (255.0 / 2)) / 255.0)
image_data = np.reshape(image_data, (1, *image_data.shape))
data = model.predict(image_data)
print(data)
print(np.argmax(data))
prediction = np.argmax(data)
CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
print(CATEGORIES[prediction])

predictions = []
for file in os.listdir('../../res/test'):
    print(file)
    cv_image = cv2.imread('../../res/test/{}'.format(file))
    cv_image = cv2.resize(cv_image, (484, 484))
    image_data = ((cv_image - (255.0 / 2)) / 255.0)
    image_data = np.reshape(image_data, (1, *image_data.shape))
    data = model.predict(image_data)
    prediction = np.argmax(data)
    predictions.append((file, CATEGORIES[prediction]))

print(predictions)

with open('submission_5.csv', 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["file", "species"])
    for row in predictions:
        writer.writerow(row)



