from math import sqrt, pi

from numpy.linalg import det, inv
import numpy as np


def pdf(mean, sigma, Xtest):
    K = 1/sqrt(((2 * pi) ** (len(mean)/2)) * det(sigma))
    return K * np.exp(-0.5 * ((Xtest - mean.T) @ inv(sigma) * (Xtest - mean.T)).sum(axis=1))

def feature_mean(samples, labels):
    distinct_labels = np.unique(labels)
    means = []
    
    for label in distinct_labels:
        means.append(samples[np.flatnonzero(labels == label)].mean(0))
        
    return np.asarray(means).T


def feature_cov(samples, labels):
    distinct_labels = np.unique(labels)
    covariances = []
    
    for label in distinct_labels:
        covariances.append(np.cov(samples[np.flatnonzero(labels == label)].T))
        
    return np.asarray(covariances)