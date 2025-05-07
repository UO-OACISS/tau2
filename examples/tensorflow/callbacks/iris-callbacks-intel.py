#https://medium.com/@nutanbhogendrasharma/tensorflow-deep-learning-model-with-iris-dataset-8ec344c49f91
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras


tf.compat.v1.keras.callbacks.TensorBoard(profile_batch=0)



tf.config.list_physical_devices()



dataFolder = 'input/'
dataFile = dataFolder + "iris.csv"
print(dataFile)

df = pd.read_csv(dataFile)
df.head()

X = df.iloc[:,0:3].values
y = df.iloc[:,4].values

print(X[0:5])
print(y[0:5])

print(X.shape)
print(y.shape)

from sklearn.preprocessing import LabelEncoder

encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)


print(y1)

Y = pd.get_dummies(y1).values
print(Y[0:5])



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


print(X_train[0:5])

print(y_train[0:5])

print(X_test[0:5])

print(y_test[0:5])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3)
  ])
model


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print("Fitting")
#model.fit(X_train, y_train, batch_size=50, epochs=100,callbacks=[tf.keras.callbacks.TensorBoard(profile_batch=0)])

callbacks = []
try:
    import tau_callbacks
    callbacks.append(tau_callbacks.TauTensorFlowCallbacks())
    print("****** Using TAU TensorFlow callbacks")
except Exception:
    print("****** Not using TAU TensorFlow callbacks")
    pass

model.fit(X_train, y_train, batch_size=50, epochs=100,callbacks=callbacks)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)



y_pred = model.predict(X_test)

y_pred

actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)

print(f"Actual: {actual}")
print(f"Predicted: {predicted}")


