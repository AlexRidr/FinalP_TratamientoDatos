<div align="center">
  <img src="https://www.cintana.com/wp-content/uploads/2021/06/UIDE-Logo-Lt.svg">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow)

# Proyecto Final - Tratamiento de Datos

## Clasificador de Carnes con Python y Jupyter Notebook

El presente trabajo tiene como objtivo crear un clasificador de imagenes de diferentes tipos de carnes de acuerdo con el repositorio proporcionado para este trabajo (https://drive.google.com/file/d/1Z5DJ-MVS1TQV1kow9mIFWTec-ZdOLRLF/view?usp=sharing)

Para el desarrollo de este clasificador se ha escogido utilizar TensorFlow y Keras.

### Tensorflow
Es una librería de código libre para Machine Learning (ML). TensorFlow permite construir y entrenar redes neuronales para detectar patrones y razonamientos usados por los humanos.

Tiene un ecosistema integral y flexible de herramientas, bibliotecas y recursos comunitarios que permite a los investigadores impulsar lo último en ML y a los desarrolladores crear e implementar fácilmente aplicaciones basadas en ML.

TensorFlow proporciona API estables de Python y C++, así como una API compatible con versiones anteriores no garantizada para otros lenguajes.

Para instalar se utiliza el siguiente comando:
```
!pip install tensorflow
```

### Keras
Keras es la API de alto nivel de la plataforma TensorFlow. Proporciona una interfaz accesible y altamente productiva para resolver problemas de aprendizaje automático (ML), con un enfoque en el aprendizaje profundo moderno. Keras cubre cada paso del flujo de trabajo del aprendizaje automático, desde el procesamiento de datos hasta el ajuste de hiperparámetros y la implementación. Fue desarrollado con un enfoque en permitir una experimentación rápida.

### Modelo Secuencial
Como sugiere su nombre, es uno de los modelos que se utiliza para investigar diversos tipos de redes neuronales donde el modelo recibe una entrada como retroalimentación y espera una salida según lo deseado. La API y la biblioteca de Keras están incorporadas con un modelo secuencial para juzgar el modelo simple completo, no el tipo de modelo complejo. Transmite los datos y fluye en orden secuencial de arriba a abajo hasta que los datos llegan al final del modelo.

La clase secuencial de Keras es una de las clases importantes como parte de todo el modelo secuencial de Keras. Esta clase ayuda a crear un clúster donde se forma un clúster con capas de información o datos que fluyen de arriba a abajo y tiene muchas capas incorporadas con _tf.Keras._ un modelo donde la mayoría de sus características se entrenan con algoritmos que proporcionan mucha secuencia al modelo.

## Código

Las librerias utilizadas para este proyecto se implementaron con las siguientes lineas de código:

```
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
```

Con el fin de obetener muestras uniformes se define el tamaño de las imagenes y se crea el data set tanto para las muestras de entrenamiento como de test utilzando el siguiente comando _tf.keras.utils.image_dataset_from_directory_ y definindo sus parametros respectivos.
Teniendo en cuenta que se tiene sets de entrenamiento y de prueba por separado no es necesario generar un subconjunto para el entrenamiento (apartado comentado).

```
batch_size = 32
img_height = 180
img_width = 180

train_df = tf.keras.utils.image_dataset_from_directory(
  "train",
  #validation_split=0.2,
  #subset="training",
  #seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_df = tf.keras.utils.image_dataset_from_directory(
  "test",
  #validation_split=0.2,
  #subset="training",
  #seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_df.class_names
print(class_names)
```
Debido a que los valores del canal RGB se encuentran en un rango [0, 255] se estandarizan estos valores para que estén en el rango [0, 1] de ta manera que sean más apropiados para la red neuronal.
```
# Estandarización de datos en el rango de [0, 1]
normalization_layer = layers.Rescaling(1./255)

normalized_df = train_df.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_df))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
```
La siguiente porción de código se enfoca en la creación del módelo Keras Sequential el cual consta de tres bloques de convolución (_tf.keras.layers.Conv2D_) con una capa de agrupación máxima (_tf.keras.layers.MaxPooling2D_) en cada uno de ellos.
```
# Creación del modelo
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
```

Se entrena el modelo durante 10 epoch con el método Keras Model.fit:
# Entrenamiento del modelo
epochs=10
history = model.fit(
  train_df,
  #validation_data=val_df,
  epochs=epochs
)


```
# Predición con los datos de test
img = tf.keras.utils.load_img(
    r"C:\Users\AlexanderVargas\Downloads\Proyecto Final\test\CLASS_06\23-CAPTURE_20220425_234738_437.png", target_size=(img_height, img_width))

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
```

A continuación se utiliza el modelo entrenado para la predicción de las imagenes en el set de test.

```
# Predición con los datos de test
img = tf.keras.utils.load_img(
    r"C:\Users\AlexanderVargas\Downloads\Proyecto Final\test\CLASS_06\23-CAPTURE_20220425_234738_437.png", target_size=(img_height, img_width))

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
```
Las siguientes porciones de código genera las matrices de confusión de los datos de entrenamiento y de prueba
```
# Obtener las etiquetas verdaderas del conjunto de entrenamiento
y_true_train = []  # Aquí se almacenan las etiquetas verdaderas del conjunto de entrenamiento

for x, y in train_df:
    y_true_train.extend(y.numpy())

# Calcular las predicciones del modelo en el conjunto de entrenamiento
y_pred_train = model.predict(train_df)
y_pred_train = np.argmax(y_pred_train, axis=1)

# Calcular la matriz de confusión para el conjunto de entrenamiento y muéstrala si lo deseas
cm_train = confusion_matrix(y_true_train, y_pred_train)
print("Matriz de Confusión (Conjunto de Entrenamiento):")
print(cm_train)
```
```
# Obtener las etiquetas verdaderas del conjunto de prueba
y_true_test = []  # Aquí se almacenan las etiquetas verdaderas del conjunto de prueba

for x, y in test_df:
    y_true_test.extend(y.numpy())

# Calcular las predicciones del modelo en el conjunto de prueba
y_pred_test = model.predict(test_df)
y_pred_test = np.argmax(y_pred_test, axis=1)

# Calcular la matriz de confusión para el conjunto de prueba y muéstrala si lo deseas
cm_test = confusion_matrix(y_true_test, y_pred_test)
print("Matriz de Confusión (Conjunto de Prueba):")
print(cm_test)
```

## Conclusiones
Con base en los datos obtenidos por la matriz de confusión se puede decir que a pesar de realizar varias pruebas con imagenes de los datos de TEST escogidos al azar se clasifican de manera correcta, la matriz de confusión muestra que es un modelo ya que la gran mayoria de datos deberian centrarse en la diagonal principal de la matriz.

## Bibliografía

*   https://github.com/tensorflow/tensorflow
*   https://www.markdowntutorial.com/
*   https://saturncloud.io/blog/how-to-create-a-confusion-matrix-for-classification-in-tensorflow/#:~:text=Creating%20a%20Confusion%20Matrix%20in%20TensorFlow&text=TensorFlow%20provides%20a%20convenient%20way,the%20predicted%20labels%20(%20y_pred%20).
*   https://www.hobbiecode.com/reconocimiento-de-imagenes-con-python/
*   https://www.educba.com/keras-sequential/
