import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = keras.Sequential([
    keras.layers.Dense(units=64, activation='relu', input_shape=(32)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=2, activation='softmax') # or 'sigmoid' for binary
])

model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy}")

predictions = model.predict(new_data)

model.save('my_classification_model.h5')
