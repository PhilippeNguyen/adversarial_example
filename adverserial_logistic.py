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
import os
from scipy.optimize import fmin_l_bfgs_b
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
fs = os.path.sep
def bfgs(input_val,input_tensor,loss,
         num_iter=5,
         maxfun=20):
    grads = K.gradients(loss,input_tensor)

    eval_func = K.function([input_tensor],[loss]+grads)
    
    
    def eval_loss_and_grads(params):
        params = params.reshape(np.shape(input_val))
        outs = eval_func([params])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
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
    for i in range(num_iter):
        print('Start of iteration', i)
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=maxfun)
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

def select_adversarial_candidates(model,X,Y,num_examples):
    predictions = model.predict(X)
    x_shape,y_shape = np.shape(X),np.shape(Y)
    new_X = np.zeros((num_examples,*x_shape[1:]))
    new_Y = np.zeros((num_examples,*y_shape[1:]))
    new_idx = 0
    
    for idx in range(x_shape[0]):
       
        if np.argmax(predictions[idx]) == np.argmax(Y[idx]):
            new_X[new_idx] = X[idx]
            new_Y[new_idx] = Y[idx]
            new_idx+=1
        if new_idx >= num_examples:
            break
    return new_X[:new_idx],new_Y[:new_idx]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', dest='dataset',
                    action='store', default='mnist012',
                    help='dataset to use (`mnist`)')
    
    parser.add_argument('--num_adversaries', dest='num_adversaries',
                action='store', default=100,type=int,
                help='number of adversaries to generate')
    parser.add_argument('--num_images', dest='num_images',
                action='store', default=3,type=int,
                help='number of images to generate')
    parser.add_argument('--output_folder', dest='output_folder',
                action='store', default=None,
                help='number of images to generate')
    
    
    args = parser.parse_args()
    if args.output_folder is not None:
        if args.output_folder.endswith(fs):
             output_folder = args.output_folder
        else:
             output_folder = args.output_folder + fs
        os.makedirs(output_folder,exist_ok=True)

    X_train,y_train,X_test,y_test = get_dataset(args.dataset)
    
    
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
    
    X_cand,y_cand = select_adversarial_candidates(model,X_train,
                                                y_train,args.num_adversaries)
    initial_predictions = model.predict(X_cand)
    unique_vectors = np.unique(y_train,axis=0)

    y_new = swap_y(y_cand,unique_vectors)
    
    y_new_tensor = K.variable(y_new)
    new_input = K.variable(X_cand)
    
    new_model = logreg_model(input_shape,num_categories,
                             input_tensor=new_input)
    new_model.set_weights(model_weights)
    loss = K.categorical_crossentropy(y_new_tensor,new_model.output)

    adversary_x = bfgs(X_cand,new_input,loss)
    adversary_y = model.predict(adversary_x)
    
    
    #check adversary predictions
    num_adversaries = np.shape(adversary_x)[0]
    successful_adv = []

    for adv_idx in range(num_adversaries):
        if np.argmax(adversary_y[adv_idx]) != np.argmax(y_cand[adv_idx]):
            successful_adv.append(adv_idx)
            if (len(successful_adv) < args.num_images 
                and plt is not None
                and args.output_folder is not None):
                plt.imshow(adversary_x[adv_idx])
                title_str = ('Predicts as {:d}, '
                             'predicted original as {:d}').format(
                                    np.argmax(adversary_y[adv_idx]),
                                    np.argmax(initial_predictions[adv_idx]))
                plt.title(title_str)
                plt.savefig(output_folder+str(adv_idx)+'.png')
    
    print("Adversaries worked : {:d} out of {:d}".format(len(successful_adv),num_adversaries))
    
    ### Test successful adversaries against a nearest neighbors classifier
    
