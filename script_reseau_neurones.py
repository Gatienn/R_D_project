from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,Input
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta
from tensorflow.keras.utils import to_categorical


def model_regression():
    model = Sequential()
    model.add(Dense(units=32,input_shape=(11,3)))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(units=15))

    sgd = SGD(learning_rate=0.1)
    model.compile(optimizer=sgd,
                loss='mean_squared_error',
                metrics=['accuracy'])
    return model


def model_classification():
    model=Sequential()
    model.add(Dense(32, input_shape=(101,3)))
    model.add(Activation("relu"))
    model.add(Dense(8))
    model.add(Activation("softmax"))

    sgd = SGD(learning_rate=0.1)
    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def taux_prediction(y_predicted,y_test):
    return sum(sum((np.argmax(y_predicted[i,j])==np.argmax(y_test[i,j])) for j in range(len(y_test[0]))) for i in range(len(y_test)))/len(y_test)


model=model_regression()
"""X1,Y1=create_dataset_sc(1000,20,10)
Y2=sc_to_class(X1,Y1)

X_train,X_test,Y_train,Y_test=train_test_split(X1,Y1,test_size=0.2,train_size=0.8)
model.fit(X_train, Y_train, epochs=100, batch_size=32)
y_predicted = model.predict(X_test)"""

