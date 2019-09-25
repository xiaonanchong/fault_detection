import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten
import numpy as np
import numpy.matlib
import random
import matplotlib.pyplot as plt

L = 2000

'''
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=L))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
'''
model = Sequential()
model.add(Conv1D(filters=512, kernel_size=10, strides=5, padding='valid', activation='relu')) #(512, 400)
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid')) #(512, 200)
model.add(Conv1D(filters=64, kernel_size=5, strides=2, padding='valid', activation='relu')) #(64, 100)
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid')) #(64, 50)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x = np.load('x.npy')
y = np.load('y.npy')
x = np.expand_dims(x, 0)
print(x.shape, y.shape)
history = model.fit(x, y, validation_split=0.33, epochs=100, batch_size=100)


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
