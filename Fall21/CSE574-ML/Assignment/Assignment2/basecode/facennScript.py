'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt, exp

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    n_train = training_data.shape[0]
    
    # add bias to training data
    training_data = np.hstack([training_data, np.ones((n_train, 1))])
    
    layer_one_net = np.dot(training_data, w1.T)
    # activation function
    layer_one_out = sigmoid(layer_one_net)
    
    # add bias to layer one
    layer_one_out = np.hstack([layer_one_out, np.ones((n_train, 1))])
    
    output_net = np.dot(layer_one_out, w2.T)
    # activation function
    output = sigmoid(output_net)
    
    # one hot encode labels
    encoded_labels = np.zeros((n_train, n_class))
    for idx in range(n_train):
        encoded_labels[idx][int(training_label[idx])] = 1.0
    
    delta = output - encoded_labels
    
    # w2 gradient: derivative of error function with respect to the weight from the hidden unit to output unit
    w2_grad = np.dot(delta.T, layer_one_out)
    
    w1_grad = layer_one_out * (1 - layer_one_out) * np.dot(delta, w2)
    w1_grad = np.dot(w1_grad.T, training_data)
    
    # remove gradient for bias term
    w1_grad = w1_grad[:-1, :]
    
    # negative log loss
    loss = -np.sum((encoded_labels * np.log(output)) + (1.0 - encoded_labels) * np.log(1.0 - output))
    
    mean_loss = loss/n_train
    
    reg = (lambdaval / 2 * n_train) * ((np.sum(w1 * w1) + np.sum(w2 * w2)))
    obj_val = loss + reg
        
    w1_grad = (w1_grad + (lambdaval * w1)) / n_train
    w2_grad = (w2_grad + (lambdaval * w2)) / n_train

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    obj_grad = np.concatenate((w1_grad.flatten(), w2_grad.flatten()), 0)

    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    n = data.shape[0]
    labels = np.zeros((n, 1))
    
    data = np.hstack([data, np.ones((n, 1))])
    
    layer_one_net = np.dot(data, w1.T)
    layer_one_out = sigmoid(layer_one_net)
    layer_one_out = np.hstack([layer_one_out, np.ones((n, 1))])
    
    output_net = np.dot(layer_one_out, w2.T)
    output = sigmoid(output_net)
    
    for i in range(n):
        labels[i] = np.argmax(output[i])

    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
