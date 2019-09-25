import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import numpy.matlib
import random
import matplotlib.pyplot as plt

L = 2000

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=L))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x = np.load('x.npy')
y = np.load('y.npy')
history = model.fit(x, y, validation_split=0.33, epochs=100, batch_size=32)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig("train_acc.png")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig("train_loss.png")