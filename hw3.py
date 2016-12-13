"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, LeNetConvPoolLayer, train_nn,LeNetConvLayer_dropout,HiddenLayer_dropout,HiddenLayer

#Problem 1
#Implement the convolutional neural network architecture depicted in HW3 problem 1
#Reference code can be found in http://deeplearning.net/tutorial/code/convolutional_mlp.py
def test_lenet(atrain_set_x, atrain_set_y, avalid_set_x, avalid_set_y, atest_set_x, atest_set_y, learning_rate=0.1, n_epochs=200, nkerns=[32, 64],nhidden=[4096,512], batch_size=600):
    rng = numpy.random.RandomState(23455)
    test_set_x, test_set_y = shared_dataset([atest_set_x, atest_set_y])
    valid_set_x, valid_set_y = shared_dataset([avalid_set_x, avalid_set_y])
    train_set_x, train_set_y = shared_dataset([atrain_set_x, atrain_set_y])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=nhidden[0],
        activation=T.tanh
    )

    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=nhidden[0],
        n_out=nhidden[1],
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=nhidden[1], n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)

############################################################################################################# #####################
#Write a function to add translations
def translate_image(x):
    M=numpy.random.random_integers(0,5)
    N=numpy.random.random_integers(0,5)
    temp=T.roll(x,M,axis=2)
    out=T.roll(temp,N,axis=3)
    return out

def translate_image_(data):
    x=numpy.reshape(data,(40000, 3, 32, 32))
    a=numpy.zeros((40000,3,32,32))
    M=numpy.random.random_integers(0,10)
    N=numpy.random.random_integers(0,10)
    a=numpy.roll(x, M, axis=2)
    a=numpy.roll(a, N, axis=3)
    output=numpy.reshape(a,(40000,3072))
    return output  

#Implement a convolutional neural network with the translation method for augmentation
def test_lenet_translation(atrain_set_x, atrain_set_y, avalid_set_x, avalid_set_y, atest_set_x, atest_set_y, learning_rate=0.1, n_epochs=200, nkerns=[32, 64],nhidden=[4096,512], batch_size=600):
    rng = numpy.random.RandomState(23455)

    test_set_x, test_set_y = shared_dataset([atest_set_x, atest_set_y])
    valid_set_x, valid_set_y = shared_dataset([avalid_set_x, avalid_set_y])
    train_set_x, train_set_y = shared_dataset([atrain_set_x, atrain_set_y])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                  # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input_pre = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=nhidden[0],
        activation=T.tanh
    )

    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=nhidden[0],
        n_out=nhidden[1],
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=nhidden[1], n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: theano.shared(numpy.asarray(translate_image_(train_set_x),dtype=theano.config.floatX),borrow=borrow)[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
###########################################################################################################################

#Problem 2.2
#Write a function to add roatations
#def rotate_image(input_x,batch_size):
#    input_x = 
##    X = T.shared(X_values, 'X')
#    print(X.eval())
#    print(X[::-1].eval())


#def rotate_image(data):
#    x=numpy.reshape(data,(40000, 3, 32, 32))
#    a=numpy.zeros((40000,3,32,32))
#    for i in range(0,40000):
#        M=numpy.random.random_integers(0,4)
#        for j in range(0,3):
#            a[i][j]=numpy.rot90(a[i][j], M)
#    output=numpy.reshape(a,(40000,3072))
#    return output
def rotate_image_(data):
    x=numpy.reshape(data,(40000, 3, 32, 32))
    a=numpy.zeros((40000,3,32,32))
    for i in range(0,40000):
        M=numpy.random.random_integers(0,4)
        for j in range(0,3):
            a[i][j]=numpy.rot90(x[i][j], M)
    output=numpy.reshape(a,(40000,3072))
    return output    
def rotate_image(data):
    x=numpy.reshape(data,(40000, 3, 32, 32))
    a=numpy.zeros((40000,3,32,32))
    for i in range(0,40000):
        M=numpy.random.random_integers(0,4)
        for j in range(0,3):
            a[i][j]=numpy.rot90(x[i][j], M)
    output=numpy.reshape(a,(40000,3072))
    return output   
    
#Implement a convolutional neural network with the rotation method for augmentation
def test_lenet_rotation(atrain_set_x, atrain_set_y, avalid_set_x, avalid_set_y, atest_set_x, atest_set_y, 
           learning_rate=0.1, n_epochs=200, nkerns=[32, 64],nhidden=[4096,512], batch_size=600):
    rng = numpy.random.RandomState(23455)
    atrain_set_x_=numpy.append(atrain_set_x, rotate_image(atrain_set_x), axis=0)
    atrain_set_y_=numpy.append(atrain_set_y,atrain_set_y, axis=0)
    test_set_x, test_set_y = shared_dataset([atest_set_x, atest_set_y])
    valid_set_x, valid_set_y = shared_dataset([avalid_set_x, avalid_set_y])
    train_set_x, train_set_y = shared_dataset([atrain_set_x_, atrain_set_y_])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                     # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    
    layer0_input = x.reshape((batch_size, 3, 32, 32))
 
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=nhidden[0],
        activation=T.tanh
    )

    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=nhidden[0],
        n_out=nhidden[1],
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=nhidden[1], n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: theano.shared(numpy.asarray(rotate_image(train_set_x),dtype=theano.config.floatX),borrow=borrow)[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
    
####################################################################################################################################
#Problem 2.3
#Write a function to flip images
def flip_image(data):
    x=numpy.reshape(data,(40000, 3, 32, 32))
    a=numpy.zeros((40000,3,32,32))
    for i in range(0,40000):
        for j in range(0,3):
            a[i][j]=numpy.fliplr(x[i][j])
    output=numpy.reshape(a,(40000,3072))
    return output    
#Implement a convolutional neural network with the flip method for augmentation
def test_lenet_flip(atrain_set_x, atrain_set_y, avalid_set_x, avalid_set_y, atest_set_x, atest_set_y, 
           learning_rate=0.1, n_epochs=200, nkerns=[32, 64],nhidden=[4096,512], batch_size=600):
    rng = numpy.random.RandomState(23455)
    atrain_set_x_=numpy.append(atrain_set_x, flip_image(atrain_set_x), axis=0)
    atrain_set_y_=numpy.append(atrain_set_y,atrain_set_y, axis=0)
    test_set_x, test_set_y = shared_dataset([atest_set_x, atest_set_y])
    valid_set_x, valid_set_y = shared_dataset([avalid_set_x, avalid_set_y])
    train_set_x, train_set_y = shared_dataset([atrain_set_x_, atrain_set_y_])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                     # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    
    layer0_input = x.reshape((batch_size, 3, 32, 32))
 
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=nhidden[0],
        activation=T.tanh
    )

    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=nhidden[0],
        n_out=nhidden[1],
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=nhidden[1], n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: theano.shared(numpy.asarray(flip_image(train_set_x),dtype=theano.config.floatX),borrow=borrow)[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
    
########################################################################################################################################### 
##Problem 2.4
#Write a function to add noise, it should at least provide Gaussian-distributed and uniform-distributed noise with zero mean
def noise_injection(data):
    x=numpy.reshape(data,(40000, 3, 32, 32))
    a=numpy.zeros((40000,3,32,32))
    for i in range(0,40000):
        for j in range(0,3):
            a[i][j]=x[i][j]+numpy.random.normal(loc=0, scale=0.06, size=(32,32))
    output=numpy.reshape(a,(40000,3072))
    return output 
    
#Implement a convolutional neural network with the augmentation of injecting noise into input
def test_lenet_inject_noise_input(atrain_set_x, atrain_set_y, avalid_set_x, avalid_set_y, atest_set_x, atest_set_y, 
           learning_rate=0.1, n_epochs=200, nkerns=[32, 64],nhidden=[4096,512], batch_size=600):
    rng = numpy.random.RandomState(23455)
    atrain_set_x_=numpy.append(atrain_set_x, noise_injection(atrain_set_x), axis=0)
    atrain_set_y_=numpy.append(atrain_set_y,atrain_set_y, axis=0)
    test_set_x, test_set_y = shared_dataset([atest_set_x, atest_set_y])
    valid_set_x, valid_set_y = shared_dataset([avalid_set_x, avalid_set_y])
    train_set_x, train_set_y = shared_dataset([atrain_set_x_, atrain_set_y_])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                     # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    
    layer0_input = x.reshape((batch_size, 3, 32, 32))
 
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=nhidden[0],
        activation=T.tanh
    )

    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=nhidden[0],
        n_out=nhidden[1],
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=nhidden[1], n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
########################################################################################################################################   
#Problem 3 
#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset

def MY_lenet(atrain_set_x, atrain_set_y, avalid_set_x, avalid_set_y, atest_set_x, atest_set_y, n_epochs=200, nkerns=[32,32,64,64],nhidden=[4096,512], batch_size=50):
    rng = numpy.random.RandomState(23455)
    atrain_set_x_=numpy.append(atrain_set_x, flip_image(atrain_set_x), axis=0)
    atrain_set_y_=numpy.append(atrain_set_y,atrain_set_y, axis=0)
    test_set_x, test_set_y = shared_dataset([atest_set_x, atest_set_y])
    valid_set_x, valid_set_y = shared_dataset([atest_set_x, atest_set_y])
    train_set_x, train_set_y = shared_dataset([atrain_set_x_, atrain_set_y_])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    training_enabled = T.iscalar('training_enabled')


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0],3, 2, 2),
        poolsize=(1,1)

    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 31, 31),
        filter_shape=(nkerns[1],nkerns[0], 3, 3),
        poolsize=(3, 3),
        stride=(2,2)
    )
    layer2 = LeNetConvLayer_dropout(
        rng,
        is_train = training_enabled,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 14, 14),
        filter_shape=(nkerns[2], nkerns[1], 2, 2)

    )
    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 13, 13),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(3,3),
        stride=(2,2)
    )
    
    layer4_input = layer3.output.flatten(2)

    layer4 = HiddenLayer_dropout(
        rng,
        is_train = training_enabled,
        input=layer4_input,
        n_in=nkerns[3] * 5 * 5,
        n_out=nhidden[0],
        activation=T.tanh,
        p = 0.7
    )

    layer5 = HiddenLayer_dropout(
        rng,
        is_train = training_enabled,
        input=layer4.output,
        n_in=nhidden[0],
        n_out=nhidden[1],
        activation=T.tanh,
        p = 0.7
    )

    # classify the values of the fully-connected sigmoidal layer
    layer6 = LogisticRegression(input=layer5.output, n_in=nhidden[1], n_out=10)

    L2_sqr = (
            (layer1.W ** 2).sum() + (layer2.W ** 2).sum() + (layer3.W ** 2).sum() + (layer4.W ** 2).sum() 
            + (layer5.W ** 2).sum() + (layer6.W ** 2).sum()
        )
    
    cost = (
        layer6.negative_log_likelihood(y)
        + 0.0001 * L2_sqr
    )

    # the cost we minimize during training is the NLL of the 
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )


    params = layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    #grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    
    #def updates(params, grads, learning_rate, momentum):
    #    updates = []
    #    memory_ = [theano.shared(numpy.zeros_like(p.get_value()))for p in params]
    #    for n, (param, grad) in enumerate(zip(params, grads)):
    #        memory = memory_[n]
    #        update = momentum * memory - learning_rate * grad
    #        update2 = momentum * momentum * memory - (
    #            1 + momentum) * learning_rate * grad
    #        updates.append((memory, update))
    #        updates.append((param, param + update2))
    #    return updates
    #updates = [
    #    (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]
    def Adam(cost, params, lr=0.02, b1=0.1, b2=0.001, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        a=0
        i = theano.shared(numpy.asarray(a,dtype=theano.config.floatX),borrow=True)
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates
    
    train_model = theano.function(
        [index],
        cost,
        updates=Adam(cost, params, lr=0.002, b1=0.1, b2=0.001, e=1e-8),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](1)
        }
    )
    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
    
'''
########################################################################################################################################
#############################################################################################################################################################################

'''