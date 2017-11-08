#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:57:05 2017

"""

import keras
from keras.datasets import mnist
import keras.backend as K
from keras.layers import (Dense,Activation,
                          Flatten,Input)
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import argparse
from scipy.optimize import fmin_l_bfgs_b


def bfgs(input_val,input_tensor,loss):
    grads = K.gradients(loss,input_tensor)

    eval_func = K.function([input_tensor],[loss]+grads)
    initial_eval = eval_func([input_val])
    
    
    def eval_loss_and_grads(params):
        params = params.reshape(np.shape(input_val))
        outs = eval_func([params])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        print(np.sum(loss_value))
        return loss_value, grad_values
    
    
    
    class Evaluator(object):
    
        def __init__(self):
            self.loss_value = None
            self.grads_values = None
    
        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value
    
        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values
        
    evaluator = Evaluator()
    
    x = input_val
    for i in range(20):
        print('Start of iteration', i)
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', np.sum(min_val))
    
    return x.reshape(np.shape(input_val))

#def sgd_momentum(input_val,input_tensor,loss):
    
def get_dataset(data_name):
    if data_name.startswith('mnist'):
        train_data,test_data = mnist.load_data()
        X_train,y_train = train_data
        X_test,y_test = test_data
        X_train = np.float32(X_train/255.)
        X_test = np.float32(X_test/255.)
        y_train = np.float32(y_train)
        y_test = np.float32(y_test)
        
        mnist_vals = data_name.replace('mnist','')
        if mnist_vals:
            mnist_ints = [int(idx) for idx in mnist_vals]
            valid_train = np.isin(y_train,mnist_ints)
            valid_test = np.isin(y_test,mnist_ints)
            
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]
            X_test = X_test[valid_test]
            y_test = y_test[valid_test]
            
        encoder = OneHotEncoder()
        encoder.fit(np.expand_dims(y_train,axis=-1))
        y_train = encoder.transform(np.expand_dims(y_train,axis=-1)).toarray()
        y_test = encoder.transform(np.expand_dims(y_test,axis=-1)).toarray()
        
    return X_train,y_train,X_test,y_test

def swap_y(y_set,unique_vectors):
    y_shape = np.shape(y_set)
    y_len,num_cat = y_shape
    
    y_new = np.zeros(y_shape)
    y_new_probs = np.argmax(np.random.multinomial(1,
                                        [1/np.float(num_cat-1)]*(num_cat-1),
                                        size=y_len),
                                        axis=1)
    for idx in range(y_len):
        not_eq = np.any(np.not_equal(y_set[idx],unique_vectors),axis=1)
        not_eq_idx = np.where(not_eq)[0]
        y_new[idx] = unique_vectors[not_eq_idx[y_new_probs[idx]]]
    
    return y_new

def logreg_model(input_shape,num_categories,
                 input_tensor=None):
    if input_tensor is None:
        inpt = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inpt = Input(tensor=input_tensor)
        else:
            inpt = input_tensor
    x = Flatten()(inpt)
    x = Dense(num_categories)(x)
    x = Activation('softmax')(x)
    model = keras.models.Model(inpt,x)
    return model
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', dest='dataset',
                    action='store', default='mnist012',
                    help='dataset to use (`mnist`)')
    
    parser.add_argument('--num_adversaries', dest='num_adversaries',
                action='store', default=32,type=int,
                help='number of adversaries to generate')
    
    args = parser.parse_args()
    
    X_train,y_train,X_test,y_test = get_dataset(args.dataset)
    
#    X_train = X_train.reshape(np.shape(X_train)[0],np.prod(np.shape(X_train)[1:]))
#    X_test = X_test.reshape(np.shape(X_test)[0],np.prod(np.shape(X_test)[1:]))

    
    input_shape = np.shape(X_train)[1:]
    num_categories = np.shape(y_train)[1]
    
    
    
    model = logreg_model(input_shape=input_shape,
                         num_categories=num_categories)
    
    model.compile('adam','categorical_crossentropy',['acc'])
    early_stop = EarlyStopping(patience=10)
    model.fit(X_train,y_train,
              epochs=100,
              callbacks=[early_stop],
              validation_data=(X_test,y_test))
    
    model_weights = model.get_weights()
    
    X_new = X_train[:args.num_adversaries]
    unique_vectors = np.unique(y_train,axis=0)

    y_old = y_train[:args.num_adversaries]
    y_new = swap_y(y_old,unique_vectors)
    y_new_tensor = K.variable(y_new)
    
    new_input = K.variable(X_new)
    
    new_model = logreg_model(input_shape,num_categories,
                             input_tensor=new_input)
    new_model.set_weights(model_weights)
    
    loss = K.categorical_crossentropy(y_new_tensor,new_model.output)

    adversary_x = bfgs(X_new,new_input,loss)

    
