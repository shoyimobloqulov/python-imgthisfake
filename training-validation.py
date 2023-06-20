import pandas as pd
import numpy as np
from functions import convert_to_ela_image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

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

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid',
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Jadval matritsasi',
                          cmap=plt.cm.Blues):
    """
        Bu funksiya chalkashlik matritsasini chop etadi va chizadi.
        Normalizatsiyani `normalize=True` o`rnatish orqali qo`llash mumkin.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Haqiqiy belgi')
    plt.xlabel('Bashoratli yorliq')

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Treningning yo'qolishi")
ax[0].plot(history.history['val_loss'], color='r', label="tasdiqlash yo'qolishi",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Treningning aniqligi")
ax[1].plot(history.history['val_accuracy'], color='r',label="Tasdiqlashning aniqligi")
legend = ax[1].legend(loc='best', shadow=True)

# Tasdiqlash ma'lumotlar to'plamidagi qiymatlarni taxmin qilish
Y_pred = model.predict(X_val)

# Bashorat sinflarini bitta issiq vektorga aylantirish
Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Tekshirish kuzatuvlarini bitta issiq vektorga aylantirish
Y_true = np.argmax(Y_val,axis = 1) 

# chalkashlik matritsasini hisoblash
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# chalkashlik matritsasini hisoblash
plot_confusion_matrix(confusion_mtx, classes = range(2))
plt.show()
