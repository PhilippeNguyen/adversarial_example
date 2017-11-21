#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 01:40:44 2017

"""

import keras
from keras.datasets import mnist
import keras.backend as K
from keras.layers import (Dense,Activation,
                          Flatten,Input)
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Optimizer
import numpy as np
import matplotlib.pyplot as plt

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
        else:
            mnist_ints =list(range(10))
            
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        
    return X_train,y_train,X_test,y_test,mnist_ints
  
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
    new_X = np.zeros((num_examples,x_shape[1],x_shape[2]))
    new_Y = np.zeros((num_examples,y_shape[1]))
    new_idx = 0
    
    for idx in range(x_shape[0]):
        if np.argmax(predictions[idx]) == np.argmax(Y[idx]):
            new_X[new_idx] = X[idx]
            new_Y[new_idx] = Y[idx]
            new_idx+=1
        if new_idx >= num_examples:
            break
    return new_X[:new_idx],new_Y[:new_idx]


class Adv_Adam(Optimizer):
  ''' Slight modification of keras Adam which returns the grad instead
      of updating as part of the tf call
  '''
  def __init__(self,
               lr=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-8,
               decay=0.,
               **kwargs):
    super(Adv_Adam, self).__init__(**kwargs)
    self.iterations = K.variable(0, name='iterations')
    self.lr = K.variable(lr, name='lr')
    self.beta_1 = K.variable(beta_1, name='beta_1')
    self.beta_2 = K.variable(beta_2, name='beta_2')
    self.epsilon = epsilon
    self.decay = K.variable(decay, name='decay')
    self.initial_decay = decay
    
  def get_updates(self, params, loss,shapes=None):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr *= (1. / (1. + self.decay * self.iterations))

    t = self.iterations + 1
    lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                 (1. - K.pow(self.beta_1, t)))
    if shapes is None:
        shapes = [K.int_shape(p) for p in params]
    ms = [K.zeros(shape) for shape in shapes]
    vs = [K.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + ms + vs

    for p, g, m, v in zip(params, grads, ms, vs):
      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
      grad= lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

      self.updates.append(K.update(m, m_t))
      self.updates.append(K.update(v, v_t))

      return grad,self.updates
  
def fit_adv(images,model,loss,
            num_iter=100,
            optimizer=None):
    if optimizer is None:
        optimizer = Adv_Adam()
        
    model_input = model.input
    grad,optimizer_updates = optimizer.get_updates(loss=[loss],
                                 params=[model_input],
                                 shapes = [images.shape])

    iter_func = K.function([model_input],
                 [loss,grad],
                 updates=optimizer_updates)
    
    for i in range(num_iter):
        loss,g = iter_func([images])
        images = images - g
        if i %10 == 0:
            print('iter ',str(i),', current adversary loss : ', loss)
            
    return images

import argparse
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
    
    #config
    data_str = args.dataset
    num_adversaries = args.num_adversaries
    
    
    X_train,y_train,X_test,y_test,mnist_ints = get_dataset(data_str)
    
    input_shape = np.shape(X_train)[1:]
    num_categories = np.shape(y_train)[1]
    
    model = logreg_model(input_shape=input_shape,
                         num_categories=num_categories)
    
    model.compile('adam','categorical_crossentropy',['acc'])
    early_stop = EarlyStopping(patience=5)
    model.fit(X_train,y_train,
              epochs=100,
              callbacks=[early_stop],
              validation_data=(X_test,y_test))
    model_weights = model.get_weights()
    
    X_cand,y_cand = select_adversarial_candidates(model,X_train,
                                                y_train,num_adversaries)
    initial_predictions = model.predict(X_cand)
    unique_vectors = np.unique(y_train,axis=0)

    y_new = swap_y(y_cand,unique_vectors)
    
    
    loss = K.sum(K.categorical_crossentropy(y_new,model.output))

    adversary_x = fit_adv(X_cand,model,loss)
    adversary_y = model.predict(adversary_x)
    
    
    
    #%%
    #check adversary predictions
    
    successful_adv = []
    for adv_idx in range(num_adversaries):
        if np.argmax(adversary_y[adv_idx]) != np.argmax(y_cand[adv_idx]):
            successful_adv.append(adv_idx)
    
    print("Adversaries worked : {:d} out of {:d}".format(len(successful_adv),num_adversaries))
    
    
    #%%
    
    successful_adv_x = adversary_x[successful_adv]
    successful_adv_y = adversary_y[successful_adv]
    original_x = X_cand[successful_adv]
    original_y = y_cand[successful_adv]
    if len(successful_adv) > 0:
        plt.imshow(successful_adv_x[0])
        print('model incorrectly predicts this adversary as ',
              mnist_ints[np.argmax(adversary_y[0])])
    if len(successful_adv) > 0:
        plt.imshow(original_x[0])
        print('model predicted the original as ' ,
              mnist_ints[np.argmax(original_y[0])])