
import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sklearn import svm, metrics
from sklearn.svm import SVC
import time

def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection
    """
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    num_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train"+str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10*n_validation

    # Construct validation data
    validation_data = np.zeros((10*n_validation, num_feature))
    for i in range(10):
        validation_data[i*n_validation:(i+1)*n_validation,
                        :] = mat.get("train"+str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10*n_validation, 1))
    for i in range(10):
        validation_label[i*n_validation:(i+1)*n_validation,
                         :] = i*np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, num_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train"+str(i)).shape[0]
        train_data[temp:temp+size_i-n_validation,
                   :] = mat.get("train"+str(i))[n_validation:size_i, :]
        train_label[temp:temp+size_i-n_validation, :] = i * \
            np.ones((size_i-n_validation, 1))
        temp = temp+size_i-n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test"+str(i)).shape[0]
    test_data = np.zeros((n_test, num_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test"+str(i)).shape[0]
        test_data[temp:temp+size_i, :] = mat.get("test"+str(i))
        test_label[temp:temp+size_i, :] = i*np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(num_feature):
        if(sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data = train_data/255.0
    validation_data = validation_data/255.0
    test_data = test_data/255.0
    return train_data, train_label, validation_data, validation_label, test_data, test_label

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def softmax(x):
    n =np.exp(x - np.max(x))
    d = np.sum(n,axis = 1)
    return n/np.expand_dims(d,1)

def blrObjFunction(params, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.
    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector
    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """

    ##################
    # YOUR CODE HERE #
    ##################
    parameters = params
    train_data, labeli = args
    num_data = train_data.shape[0]
    num_feature = train_data.shape[1]
    error_grad = np.zeros((num_feature+1, 1))

    # bias added to front of train_data
    train_data = np.hstack((np.ones((train_data.shape[0], 1)), train_data))
    parameters = np.matrix(parameters)
    parameters = parameters.T

    # duplicate weight vector to 50000,716
    parameters = np.reshape(parameters, (716, 1))

    # calculate Y
    Y = sigmoid(np.dot(train_data, parameters))

    d = np.multiply(labeli, np.log(Y)) + \
        np.multiply((1.0 - labeli), np.log(1.0 - Y))
    error = np.sum(d)
    error = -error
    t = np.multiply((Y - labeli), train_data)
    grad_error = np.sum(t, axis=0)
    grad_error = np.squeeze(np.asarray(grad_error))
    return error, grad_error
    return error, grad_error

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix
    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################

    # Bias added to the front of train_data
    bias = np.ones((data.shape[0], 1))
    data = np.hstack((bias, data))

    # np.argmax of returned value by sigmoid is the label
    b = sigmoid(np.dot(data, W))
    label = np.argmax(b, 1)
    label.resize((data.shape[0], 1))
    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.
    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector
    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    num_data = train_data.shape[0]
    num_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((num_feature + 1, n_class))
    error_grad.flatten()

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    x = np.hstack((np.ones((num_data, 1)), train_data))
    W = params.reshape((num_feature + 1, n_class))
    theta = softmax(np.dot(x, W))
    temp = T * np.log(theta)
    error = -(np.sum(temp))
    error_grad = np.dot(x.T, theta - labeli).ravel()

    return error, error_grad

def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix
    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
   
    x = np.hstack((np.ones((data.shape[0],1)), data))
    theta = softmax(np.dot(x,W))
    label = np.argmax(theta, axis = 1).reshape((data.shape[0], 1))
    return label

"""
<br>
Script for Logistic Regression<br>
"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

"""number of classes"""

n_class = 10

"""number of training samples"""

n_train = train_data.shape[0]

"""number of features"""

num_feature = train_data.shape[1]

T = np.zeros((n_train, n_class))
for i in range(n_class):
    T[:, i] = (train_label == i).astype(int).ravel()

"""Logistic Regression with Gradient Descent"""

W = np.zeros((num_feature+1, n_class))
initialWeights = np.zeros((num_feature+1, 1))
opts = {'maxiter': 50}
# for i in range(n_class):
#     # print(i)
#     labeli = T[:, i].reshape(n_train, 1)
#     args = (train_data, labeli)
#     nn_params = minimize(blrObjFunction, initialWeights,
#                          jac=True, args=args, method='CG', options=opts)
#     W[:, i] = nn_params.x.reshape((num_feature+1,))

# pickle_file = open('params.pickle', 'wb')
# pickle.dump([W], pickle_file)
# pickle_file.close()

"""Find the accuracy on Training Dataset"""

# predicted_label = blrPredict(W, train_data)
# #print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
# correct = 0
# for i in range(predicted_label.shape[0]):
#     if predicted_label[i] == train_label[i]:
#         correct += 1
# print('\n Training set Accuracy' +
#       str((correct/predicted_label.shape[0]) * 100))

# """Find the accuracy on Validation Dataset"""

# predicted_label = blrPredict(W, validation_data)
# print('\n Validation set Accuracy:' + str(100 *
#       np.mean((predicted_label == validation_label).astype(float))) + '%')

# """Find the accuracy on Testing Dataset"""

# predicted_label = blrPredict(W, test_data)
# print('\n Testing set Accuracy:' +
#       str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

"""
<br>
Script for Support Vector Machine<br>
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

train_label = train_label.reshape(train_label.shape[0])
test_label = test_label.reshape(test_label.shape[0])
validation_label = validation_label.reshape(validation_label.shape[0])

# am = SVC(kernel='linear')
# am.fit(train_data, train_label)
# print('\n\n Accuracy using linear kernel and all other parameters as default\n\n')
# print('\n Training set Accuracy: ' +
#       str(am.score(train_data, train_label)*100) + '%')
# print('\n Validation set Accuracy: ' +
#       str(am.score(validation_data, validation_label)*100) + '%')
# print('\n Testing set Accuracy: ' +
#       str(am.score(test_data, test_label)*100) + '%')

# am = SVC(kernel='rbf', gamma=1.0)
# am.fit(train_data, train_label)
# print('\n\n Accuracy using rbf kernel and gamma value 1 and all other parameters as default\n\n')
# print('\n Training set Accuracy: ' +
#       str(am.score(train_data, train_label)*100) + '%')
# print('\n Validation set Accuracy: ' +
#       str(am.score(validation_data, validation_label)*100) + '%')
# print('\n Testing set Accuracy: ' +
#       str(am.score(test_data, test_label)*100) + '%')

# am = SVC(kernel='rbf')
# am.fit(train_data, train_label)
# print('\n\n Accuracy using rbf kernel and gamma value default and all other parameters as default\n\n')
# print('\n Training set Accuracy: ' +
#       str(am.score(train_data, train_label)*100) + '%')
# print('\n Validation set Accuracy: ' +
#       str(am.score(validation_data, validation_label)*100) + '%')
# print('\n Testing set Accuracy: ' +
#       str(am.score(test_data, test_label)*100) + '%')

# am = SVC(kernel='rbf', C=1)
# am.fit(train_data, train_label)
# print('\n\n Accuracy using rbf kernel and gamma value default and c=1 and all other parameters as default\n\n')
# print('\n Training set Accuracy: ' +
#       str(am.score(train_data, train_label)*100) + '%')
# print('\n Validation set Accuracy: ' +
#       str(am.score(validation_data, validation_label)*100) + '%')
# print('\n Testing set Accuracy: ' +
#       str(am.score(test_data, test_label)*100) + '%')

# for i in range(10, 110, 10):
#     am = SVC(kernel='rbf', C=i)
#     am.fit(train_data, train_label)
#     print('\n\n Accuracy using rbf kernel and gamma value default and C value = ', i, '\n\n')
#     print('\n Training set Accuracy: ' +
#           str(am.score(train_data, train_label)*100) + '%')
#     print('\n Validation set Accuracy: ' +
#           str(am.score(validation_data, validation_label)*100) + '%')
#     print('\n Testing set Accuracy: ' +
#           str(am.score(test_data, test_label)*100) + '%')

### FOR EXTRA CREDIT

Wb = np.zeros((num_feature + 1, n_class))
initial_weights_b = np.zeros((num_feature + 1, n_class))
b_opts = {'maxiter': 100}

b_args = (train_data, T)
nn_params = minimize(mlrObjFunction, initial_weights_b,
                    jac=True, args=b_args, method='CG', options=b_opts)
Wb = nn_params.x.reshape((num_feature + 1, n_class))

# Find the accuracy on Training Dataset
b_predicted_label = mlrPredict(Wb, train_data)
print('\n Training set Accuracy:' + str(100 *
     np.mean((b_predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
b_predicted_label = mlrPredict(Wb, validation_data)
print('\n Validation set Accuracy:' + str(100 *
     np.mean((b_predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
b_predicted_label = mlrPredict(Wb, test_data)
print('\n Testing set Accuracy:' + str(100 *
     np.mean((b_predicted_label == test_label).astype(float))) + '%')

