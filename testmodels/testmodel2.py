import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1) Data: MNIST (10 classes)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0  # scale to [0,1]
x_test  = x_test.astype("float32")  / 255.0

# 2) Model
model = keras.Sequential([
    layers.Input(shape=(28, 28)),   # (H,W) for grayscale images
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")   # num_classes = 10
])

# 3) Compile
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),  # integer labels
    metrics=["accuracy"]
)

# 4) Train with EarlyStopping
early_stop = keras.callbacks.EarlyStopping(
    patience=3, restore_best_weights=True, monitor="val_accuracy"
)
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 5) Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# 6) Save & predict
model.save("mnist_classifier.keras")
probs = model.predict(x_test[:5])
preds = probs.argmax(axis=1)
print("Predicted classes:", preds)

