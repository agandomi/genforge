# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
from .SoftmaxModel import SoftmaxModel

def gp_evaluate_softmax_custom(args):
    
    """
    Train a one-layer softmax function using Keras and evaluate the probabilities and losses.
    """
    
    ytr, yval, yts, ybin_tr, ybin_val, ybin_ts, num_class, \
        optimizer_type, optimizer_param, initializer,\
        regularization, regularization_rate, batch_size, epochs,\
        patience, random_seed, buffer_size, shuffle, verbose, id_pop, id_ind,\
        gene_out_tr, gene_out_val, gene_out_ts = args
    
    # Setting Softmax Model parameters
    params = {
        'xtrain': gene_out_tr,
        'ytrain': ytr,
        'yonehot_train': ybin_tr,
        'xval': gene_out_val,
        'yval': yval,
        'yonehot_val': ybin_val,
        'num_class': num_class,
        'optimizer': optimizer_type,
        'optimizer_param': optimizer_param,
        'initializer': initializer,
        'regularization': regularization,
        'regularization_rate': regularization_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'patience': patience,
        'random_seed': random_seed,
        'buffer_size': buffer_size,
        'shuffle': shuffle,
        'verbose': verbose,
        }
    # Comiling the model
    model = SoftmaxModel.compiler(**params)
    model.fit()
    
    # Predict probabilities on the training data
    prob_tr = model.predict(gene_out_tr)
    
    # t5 = time.time()
    # print(f'Predict Time: {t5-t44}')
    if gene_out_val is not None:
        prob_val = model.predict(gene_out_val)
    else:
        prob_val = None
    
    if gene_out_ts is not None:
        prob_ts = model.predict(gene_out_ts)
    else:
        prob_ts = None
    
    # Calculate the loss on the training data
    loss_tr = model.loss(prob_tr, ybin_tr)
    
    # Calculate the loss on the validation and testing data
    if gene_out_val is not None:
        loss_val = model.loss(prob_val, ybin_val)
    else:
        loss_val = None
    
    if gene_out_ts is not None:
        loss_ts = model.loss(prob_ts, ybin_ts)
    else:
        loss_ts = None
    
    # calculate predicted labels
    yp_tr = np.argmax(prob_tr, axis=1)
    
    if gene_out_val is not None:
        yp_val = np.argmax(prob_val, axis=1)
    else:
        yp_val = None
    
    if gene_out_ts is not None:
        yp_ts = np.argmax(prob_ts, axis=1)
    else:
        yp_ts = None
    
    # Extract weights and biases
    weights = model.weight
    biases = model.bias
    weights_with_biases = np.concatenate([weights, biases], axis=1)  # Add biases as the last column
    
    # Setting the results
    results = [prob_tr, prob_val, prob_ts, loss_tr, loss_val, loss_ts, yp_tr, yp_val, yp_ts, weights_with_biases]
        
    return results
