import cv2
from keras.models import load_model
import numpy as np


# Cargar el modelo entrenado
modelo = load_model('modelo.h5')

def clasificar_documento(imagen_path):
    # Cargar la imagen del documento
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    # Reescalar la imagen a 28x28 píxeles
    imagen = cv2.resize(imagen, (28, 28))
    # Normalizar la imagen
    imagen = imagen.astype('float32') / 255
    imagen = np.expand_dims(imagen, axis=-1)
    # Predecir la clase utilizando el modelo entrenado
    resultado = modelo.predict(np.array([imagen]))
    # Obtener la clase con la probabilidad más alta
    clase_predicha = np.argmax(resultado)
    return clase_predicha

# Ejemplo de clasificación de un documento
documento_path = r'C:\Users\oscar.boullosa\Downloads\nueve.png'
clase_predicha = clasificar_documento(documento_path)

print(f'La clase predicha para el documento es: {clase_predicha}')
