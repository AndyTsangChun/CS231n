import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']
    # FP to first layer affine-relu
    a1, cache = affine_relu_forward(X, W1, b1)
    # FP to second layer affine
    scores, cache2 = affine_forward(a1, W2, b2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # softmax
    sm_loss, sm_dx = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))
    loss = sm_loss + reg_loss
    
    # BP to second layer
    dx2, dW2, db2 = affine_backward(sm_dx, cache2)
    dW2 += self.reg * W2
    # BP to first layer
    dx1, dW1, db1 = affine_relu_backward(dx2, cache)
    dW1 += self.reg * W1

    grads.update({'W1': dW1,
                  'b1': db1,
                  'W2': dW2,
                  'b2': db2})
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    self.L = len(hidden_dims) + 1
    all_dims = [input_dim] + hidden_dims + [num_classes] #Mix all dimension number into an array
    #For instance [3072, 100, 100, 10], will generate 3 sets of weight
    #1st layer = 3072*100, 2nd layer = 100*100, 3rd layer = 100*10, output layer don't have weights
    weights = {'w' + str(i+1):
               weight_scale * np.random.randn(all_dims[i], all_dims[i + 1]) 
               for i in range(len(all_dims) -1)}
    #Similar as weight for bias
    bias = {'b' + str(i + 1): np.zeros(all_dims[i + 1])
             for i in range(len(all_dims) - 1)}
    self.params.update(bias)
    self.params.update(weights)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
        self.bn_params = {'bn_param' + str(i + 1): {'mode': 'train',
                                                    'running_mean': np.zeros(all_dims[i + 1]),
                                                    'running_var': np.zeros(all_dims[i + 1])}
                          for i in xrange(len(all_dims) - 2)}
        gammas = {'gamma' + str(i + 1):
                  np.ones(all_dims[i + 1]) for i in range(len(all_dims) - 2)}
        betas = {'beta' + str(i + 1): np.zeros(all_dims[i + 1])
                 for i in range(len(all_dims) - 2)}
        self.params.update(betas)
        self.params.update(gammas)
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for key, bn_param in self.bn_params.iteritems():
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    nn_layers = {}
    # input layers
    nn_layers['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    if self.use_dropout:
        # dropout on the input layer no dropout and batchnorm needed
        drop_loss, drop_cache = dropout_forward(nn_layers['h0'], self.dropout_param)
        nn_layers['hdrop0'] = drop_loss
        nn_layers['cache_hdrop0'] = drop_cache
    # for the rest hidden layers
    for i in range(self.L):
        index = i + 1
        w = self.params['w' + str(index)]
        b = self.params['b' + str(index)]
        h = nn_layers['h' + str(index - 1)]
        if self.use_dropout:
            h = nn_layers['hdrop' + str(index - 1)]
        if self.use_batchnorm and index != self.L:
            gamma = self.params['gamma' + str(index)]
            beta = self.params['beta' + str(index)]
            bn_param = self.bn_params['bn_param' + str(index)]
            
        # FP for last layer
        if index == self.L:
            last_loss, last_cache = affine_forward(h, w, b)
            nn_layers['h' + str(index)] = last_loss
            nn_layers['cache_h' + str(index)] = last_cache
        # All other layers
        else:
            if self.use_batchnorm:
                layer_loss, layer_cache = affine_norm_relu_forward(h, w, b, gamma, beta, bn_param)
                nn_layers['h' + str(index)] = layer_loss
                nn_layers['cache_h' + str(index)] = layer_cache
            else:
                 layer_loss, layer_cache = affine_relu_forward(h, w, b)
                 nn_layers['h' + str(index)] = layer_loss
                 nn_layers['cache_h' + str(index)] = layer_cache
            # If using dropout
            if self.use_dropout:
                h = nn_layers['h' + str(index)]
                drop_loss, drop_cache = dropout_forward(h, self.dropout_param)
                nn_layers['hdrop' + str(index)] = drop_loss
                nn_layers['cache_hdrop' + str(index)] = drop_cache
    scores = nn_layers['h' + str(self.L)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # Calculating loss
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0
    #
    for w in [self.params[f] for f in self.params.keys() if f[0] == 'w']:
        reg_loss += 0.5 * self.reg * np.sum(w**2)
    loss = data_loss + reg_loss
    # BP
    nn_layers['dh' + str(self.L)] = dscores
    for i in range(self.L)[::-1]:
        index = i + 1
        dh = nn_layers['dh' + str(index)]
        cache_h = nn_layers['cache_h' + str(index)]
        #BP for last layer no dropout and batchnorm needed
        if index == self.L:
            dh, dw, db = affine_backward(dh, cache_h)
            nn_layers['dh' + str(index - 1)] = dh
            nn_layers['dw' + str(index)] = dw
            nn_layers['db' + str(index)] = db
        #BP for other layers, normal, dropout & batchnorm
        else:
            if self.use_dropout:
                cache_hdrop = nn_layers['cache_hdrop' + str(index)]
                dh = dropout_backward(dh, cache_hdrop)
            if self.use_batchnorm:
                dh, dw, db, dgamma, dbeta = affine_norm_relu_backward(dh, cache_h)
                nn_layers['dh' + str(index - 1)] = dh
                nn_layers['dw' + str(index)] = dw
                nn_layers['db' + str(index)] = db
                nn_layers['dgamma' + str(index)] = dgamma
                nn_layers['dbeta' + str(index)] = dbeta
            else:
                dh, dw, db = affine_relu_backward(dh, cache_h)
                nn_layers['dh' + str(index - 1)] = dh
                nn_layers['dw' + str(index)] = dw
                nn_layers['db' + str(index)] = db  
        # Get all weights gradient
        list_dw = {key[1:]: val + self.reg * self.params[key[1:]]
                   for key, val in nn_layers.iteritems() if key[:2] == 'dw'}
        # Get all bias gradient
        list_db = {key[1:]: val for key, val in nn_layers.iteritems() if key[:2] == 'db'}
        # Get all gamma gradient
        list_dgamma = {key[1:]: val for key, val in nn_layers.iteritems() if key[:6] == 'dgamma'}
        # Get all beta gradient
        list_dbeta = {key[1:]: val for key, val in nn_layers.iteritems() if key[:5] == 'dbeta'}
        
        grads = {}
        grads.update(list_dw)
        grads.update(list_db)
        grads.update(list_dgamma)
        grads.update(list_dbeta)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU and batchnorm

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  #Structure FP-BN-ReLU
  #Perform a regular forward pass first
  h, h_cache = affine_forward(x, w, b)
  hnorm, hnorm_cache = batchnorm_forward(h, gamma, beta, bn_param)
  hnormrelu, relu_cache = relu_forward(hnorm)
  cache = (h_cache, hnorm_cache, relu_cache)

  return hnormrelu, cache

def affine_norm_relu_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  h_cache, hnorm_cache, relu_cache = cache

  dhnormrelu = relu_backward(dout, relu_cache)
  dhnorm, dgamma, dbeta = batchnorm_backward_alt(dhnormrelu, hnorm_cache)
  dx, dw, db = affine_backward(dhnorm, h_cache)

  return dx, dw, db, dgamma, dbeta






