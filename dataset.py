import pandas as pd
import numpy as np
from functions import convert_to_ela_image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

optimizer = RMSprop(learning_rate=0.001)

dataset = pd.read_csv('datasets/dataset.csv')

X = []
Y = []

for index, row in dataset.iterrows():
    X.append(np.array(convert_to_ela_image(row[1], 90).resize((128, 128))).flatten() / 255.0)
    Y.append(row[2])

X = np.array(X)
Y = to_categorical(Y, 2)

X = X.reshape(-1, 128, 128, 3)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid',
                 activation='relu', input_shape=(128, 128, 3)))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid',
                 activation='relu'))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.summary()

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
early_stopping = EarlyStopping(monitor='val_accuracy',
                               min_delta=0,
                               patience=31,
                               verbose=0, mode='auto')

epochs = 30
batch_size = 100

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, Y_val), verbose=2, callbacks=[early_stopping])
