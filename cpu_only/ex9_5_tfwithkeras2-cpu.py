# set to use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['KERAS_BACKEND'] = 'theano'

import tensorflow as tf
from tensorflow import keras
print(keras.__version__)

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

class DNN(Model):
    def __init__(self, Nin=2, Nh_l=[2,2], Nout=2):
        super(DNN, self).__init__()
        self.dense1 = Dense(Nh_l[0], activation='relu')
        self.dense2 = Dense(Nh_l[1], activation='relu')
        self.dense3 = Dense(Nout, activation='softmax')
        
    def call(self, x):
        x = self.dense1(x)
        x = Dropout(0.5)(x)
        x = self.dense2(x)
        x = Dropout(0.25)(x)
        return self.dense3(x)
    
import numpy as np
from tensorflow.keras import datasets  # mnist

def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    L, H, W = X_train.shape
    X_train = X_train.reshape(-1, H * W)
    X_test = X_test.reshape(-1, H * W)

    X_train = (X_train / 255.0).astype(np.float32)
    X_test = (X_test / 255.0).astype(np.float32)

    return (X_train, y_train), (X_test, y_test)

cost_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = keras.optimizers.Adam()
tr_loss = keras.metrics.Mean(name='train_loss')
te_loss = keras.metrics.Mean(name='test_loss')
tr_acc = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
te_acc = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

def train(model, X_train, Y_train, N_tr, batch_size):
    for b in range(N_tr // batch_size):
        X_tr_b = X_train[batch_size * (b-1):batch_size * b]
        Y_tr_b = Y_train[batch_size * (b-1):batch_size * b]
        with tf.GradientTape() as tape:
            pred = model(X_tr_b, training=True)
            cost = cost_fn(Y_tr_b, pred)
        grad = tape.gradient(cost, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        tr_loss(cost)
        tr_acc(Y_tr_b, pred)

def validation(model, X_test, Y_test):
    pred = model(X_test, training=False)
    cost = cost_fn(Y_test, pred)
    te_loss(cost)
    te_acc(Y_test, pred)
    
from keraspp.skeras import plot_loss, plot_acc
import matplotlib.pyplot as plt

def main(epochs=20):
    Nin = 784
    Nh_l = [100, 50]
    number_of_class = 10
    Nout = number_of_class

    data = Data_func()
    model = DNN(Nin, Nh_l, Nout)
    batch_size = 100
    (X_train, Y_train), (X_test, Y_test) = data
    N_tr = X_train.shape[0]
    
    loss_l = {"loss":[], "val_loss":[]}
    acc_l = {"accuracy":[], "val_accuracy":[]}
    for epoch in range(epochs):
        # Train
        train(model, X_train, Y_train, N_tr, batch_size)     
        
        # Validation
        validation(model, X_test, Y_test)

        print(
            f'Epoch {epoch}, '
            f'Loss: {tr_loss.result():.3}, '
            f'Acc: {tr_acc.result() * 100:.3}, '
            f'Test Loss: {te_loss.result():.3}, '
            f'Test Accuracy: {te_acc.result() * 100:.3}')
        
        loss_l["loss"].append(tr_loss.result())
        acc_l["accuracy"].append(tr_acc.result())
        loss_l["val_loss"].append(te_loss.result())
        acc_l["val_accuracy"].append(te_acc.result())
        
    plot_loss(loss_l)
    plot_acc(acc_l)
    plt.show()     

if __name__ == '__main__': 
    main()