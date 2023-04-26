import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, add, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def draw_curves(history, key1='accuracy', ylim1=(0.7, 1.00)):
    plt.figure(figsize=(12, 4))
    plt.plot(history.history[key1], "r--")
    plt.plot(history.history['val_' + key1], "g--")
    plt.ylabel(key1)
    plt.xlabel('Epoch')
    plt.ylim(ylim1)
    plt.legend(['train', 'test'], loc='best')
    plt.show()

#wczytanie danych
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_val, y_val) = cifar10.load_data()
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
class_names = ['samolot', 'samochód', 'ptak', 'kot', 'jeleń',
               'pies', 'żaba', 'koń', 'statek', 'ciężarówka']
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train_cnn = X_train.reshape((X_train.shape[0], 32, 32, 3))
X_test_cnn = X_test.reshape((X_test.shape[0], 32, 32, 3))


EarlyStop = EarlyStopping(monitor='val_loss',
                          patience=5,
                          verbose=1)

#model uczenia
model = Sequential([
    Conv2D(filters=128, kernel_size=(5,5), input_shape=(32,32,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=128, kernel_size=(5,5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2,2)),
    Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2,2)),
    Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2,2)),
    Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Flatten(),
    Dense(units=32, activation="relu"),
    Dropout(0.15),
    Dense(units=16, activation="relu"),
    Dropout(0.05),
    Dense(units=10, activation="softmax")
])
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train_cnn,
                    y_train,
                    epochs=100,
                    verbose=1,
                    validation_data = (X_test_cnn, y_test),
                    callbacks = [EarlyStop]
                   )
draw_curves(history, key1='accuracy') 
score = model.evaluate(X_val, y_val, verbose=0)
print("CNN Error: %.2f%%" % (100 - score[1] * 100))