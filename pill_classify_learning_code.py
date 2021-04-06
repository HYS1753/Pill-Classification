from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf

caltech_dir = "/home/xxx/pill"	# xxx : image dir
categories = ["acertil", "actos", "amaryl", "amarylm", "amodipin",
              "capril", "diabex", "diamicron", "dilatrend", "duvie",
              "glupa", "januvia", "nimodipine", "norvasc", "olmetec",
              "sevikar", "telmisartan", "trajenta", "valsaone", "zemiglo"]
nb_classes = len(categories)

image_w = 300
image_h = 400

pixels = image_h * image_w * 3

x = []
y = []

for idx, cat in enumerate(categories):
    #one-hot encoding
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")

    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        x.append(data)
        y.append(label)
        if i % 700 == 0:
            print(cat, ":", f)

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y)
xy = (x_train, x_test, y_train, y_test)

print(x_train.shape)
print(x_train.shape[0])

#일반화
x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 300, 3))),
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(20, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=50, epochs=10, validation_data=(x_test, y_test))

print("정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_weights.h5")
print("Saved model to disk")
