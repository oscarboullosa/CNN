import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model

# Carga de datos
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalización de imagenes
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(f'Tamaño del conjunto de entrenamiento: {len(X_train)}')
print(f'Tamaño del conjunto de test: {len(X_test)}')

imagenes, etiquetas = X_train[:2], Y_train[:2]
print(f'Tamñaño de las imágenes: {imagenes.shape[1:]}')

fig, axs = plt.subplots(1, 2, figsize=(10, 10))
for i in range(2):
    axs[i].imshow(np.squeeze(imagenes[i]), cmap='gray')
    axs[i].set_title(f'Etiqueta: {etiquetas[i]}')
    axs[i].axis('off')

plt.show()

# Definimos el modelo
modelo = Sequential()
modelo.add(Conv2D(filters=6, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28, 1)))
modelo.add(MaxPooling2D(pool_size=2))
modelo.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))
modelo.add(MaxPooling2D(pool_size=2))

# Bloque clasificador de características
modelo.add(Flatten())
modelo.add(Dense(units=120, activation='relu'))
modelo.add(Dense(units=84, activation='relu'))
modelo.add(Dense(units=10, activation='softmax'))

print(modelo.summary())

#Compilación CNN
modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

historia=modelo.fit(X_train,Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=30, batch_size=64, verbose=2)

plt.plot(historia.history['accuracy'])
plt.plot(historia.history['val_accuracy'])
plt.xlabel('Época')
plt.ylabel('Exactitud (%)')
plt.legend(['Entrenamiento','Test'])
plt.show()

modelo.save('modelo.h5')
