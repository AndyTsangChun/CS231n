import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0] #500
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    f = X[i].dot(W) #score of each class (C,)
    f -= np.max(f) #From cs231n notes, shift the whole f b4 exp, prevent blowup
    exp_f = np.exp(f) #exp all of them
    correct_class_f = exp_f[y[i]] #y[i] = label of ith example, return f of the correct class
    prob = correct_class_f / np.sum(exp_f) #normalise all class
    loss += -np.log10(prob) #Loss function of Softmax
    #Calcuat dW
    for c in range(num_classes):
        prob_i = exp_f[c] / np.sum(exp_f)
        dW[:,c] += (prob_i - (c==y[i])) * X[i]
    
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  f -= np.max(f) #From cs231n notes, shift the whole f b4 exp, prevent blowup
  f = np.exp(f) #exp all of them
  sum_f = np.sum(f,1)
  correct_class_f = f[np.arange(num_train),y] 
  prob = correct_class_f / sum_f
  loss = np.sum(-np.log(prob))
  #Calcuat dW
  prob_all = (f.T / sum_f).T #Calculate prob of all training against all classes
  prob_all[np.arange(num_train),y] -= 1 #For correct class of each example -1
  dW = X.T.dot(prob_all)

  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

