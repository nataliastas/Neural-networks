#import potrzebnych bibliotek
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.optimizers import SGD,RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, add, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Convolution2D, MaxPool2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

#funkcja do rysowania wykresów dla accuracy i funkcji straty
def draw_curves(history, key1, ylim1):
    plt.figure(figsize=(12, 4))
    plt.plot(history.history[key1], "r--")
    plt.plot(history.history['val_' + key1], "g--")
    plt.ylabel(key1)
    plt.xlabel('Epoch')
    plt.ylim(ylim1)
    plt.legend(['train', 'test'], loc='best')
    plt.show()

#wczytanie danych i przygotowanie ich do procesu uczenia
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_val, y_val) = cifar10.load_data()
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
class_names = ['samolot', 'samochód', 'ptak', 'kot', 'jeleń',
               'pies', 'żaba', 'koń', 'statek', 'ciężarówka']

#implementacja modelu sieci neuronowej
model = Sequential([
    Convolution2D(filters=128, kernel_size=(5,5), input_shape=(32,32,3), activation='relu', padding='same'),
    BatchNormalization(),
    Convolution2D(filters=128, kernel_size=(5,5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2,2)),
    Convolution2D(filters=64, kernel_size=(5,5), activation='relu', padding='same'),
    BatchNormalization(),
    Convolution2D(filters=64, kernel_size=(5,5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2,2)),
    Convolution2D(filters=32, kernel_size=(5,5), activation='relu', padding='same'),
    BatchNormalization(),
    Convolution2D(filters=32, kernel_size=(5,5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2,2)),
    Convolution2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Convolution2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Flatten(),
    Dense(units=32, activation="relu"),
    Dropout(0.15),
    Dense(units=16, activation="relu"),
    Dropout(0.05),
    Dense(units=10, activation="softmax")
])
#implementacja otymalizatora, funkcji straty oraz metryki
optim = RMSprop(lr=0.001)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#uczenie przygotowanego przez nas modelu na danych treningowych i walidacyjnych
history = model.fit(
   X_train,
   to_categorical(y_train),
   epochs=80,
   validation_split=0.15,
   verbose=1
)
#sprawdzenie skuteczności wytrenowanego algorytmu na danych testowych
eval = model.evaluate(X_test, to_categorical(y_test))
eval

#wywołanie funkcji do rysowania wykresu dokładności względem epok
draw_curves(history, key1='accuracy',ylim1=(0.7, 1.00))

#wywołanie funkcji do rysowania wykresu funkcji straty względem epok
draw_curves(history, key1='loss',ylim1=(0.00, 2.50))

#wizualizacja filtrów
for layer in model.layers:
    if 'conv' in layer.name:
        weights, bias= layer.get_weights()
        f_min, f_max = weights.min(), weights.max()
        filters = (weights - f_min) / (f_max - f_min)
        print(layer.name, filters.shape)
        filter_cnt=1
        for i in range(filters.shape[3]):
            filt=filters[:,:,:, i]
            for j in range(filters.shape[0]):
                ax= plt.subplot(filters.shape[3], filters.shape[0], filter_cnt  )
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(filt[:, j])
                filter_cnt+=1
        plt.show()

#wizualizacja map cech
img_path='C:/Users/natal/Desktop/plane.jpg'
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
img = load_img(img_path, target_size=(32, 32))
x   = img_to_array(img)
x   = x.reshape((1,) + x.shape)
x /= 255.0
successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  print(feature_map.shape)
  if len(feature_map.shape) == 4:
    n_features = feature_map.shape[-1]
    size       = feature_map.shape[ 1]
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x
    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' )