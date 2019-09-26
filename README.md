# fault_detection
![accuracy](https://github.com/xiaonanchong/fault_detection/blob/master/train_acc.png)
![loss](https://github.com/xiaonanchong/fault_detection/blob/master/train_loss.png)


model = Sequential()
model.add(Conv1D(filters=1024, kernel_size=4, strides=1, padding='valid', activation='relu')) 
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid')) 
model.add(Conv1D(filters=512, kernel_size=8, strides=2, padding='valid', activation='relu')) 
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid')) 
model.add(Conv1D(filters=64, kernel_size=16, strides=4, padding='valid', activation='relu')) 
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid')) 
model.add(Flatten())
model.add(Dense(1), activation='sigmoid')
669/669 [==============================] - 336s 503ms/step - loss: 7.0664 - accuracy: 0.5007 - val_loss: 8.1808 - val_accuracy: 0.4924
Epoch 2/100  
669/669 [==============================] - 343s 513ms/step - loss: 7.9988 - accuracy: 0.5037 - val_loss: 8.1808 - val_accuracy: 0.4924
Epoch 3/100  
