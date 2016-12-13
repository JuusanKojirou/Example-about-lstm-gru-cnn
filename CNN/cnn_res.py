import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from cnn_utils import shared_dataset, load_data
from cnn_nn import LogisticRegression, LeNetConvPoolLayer, train_nn,pooling_LeNetConvPoolLayer,upsampling_LeNetConvPoolLayer,drop


def MY_CNN(atrain_set_x, atrain_set_y, avalid_set_x, avalid_set_y, atest_set_x, atest_set_y, 
           learning_rate=0.1, n_epochs=200, batch_size=1):
    rng = numpy.random.RandomState(23455)

    test_set_x, test_set_y = shared_dataset([atest_set_x[0:99], atest_set_y[0:99]])
    valid_set_x, valid_set_y = shared_dataset([atest_set_x[0:99], atest_set_y[0:99]])
    train_set_x, train_set_y = shared_dataset([atrain_set_x, atrain_set_y])

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
    y = T.matrix('y')  

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

 
    
    drop_layer=drop(x,p=0.7)
    layer0_input = drop_layer.reshape((batch_size, 3, 32, 32))
    
 

    layer11 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(64, 3, 3, 3),
        poolsize=(1, 1)
    )

    layer11_ = LeNetConvPoolLayer(
        rng,
        input=layer11.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
        poolsize=(1, 1)
    )
    

    layer21 = pooling_LeNetConvPoolLayer(
        rng,
        input=layer11_.output,
        filter_shape=(128, 64, 3, 3),
        poolsize=(2, 2)
    )
    layer21_ = LeNetConvPoolLayer(
        rng,
        input=layer21.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),
        poolsize=(1, 1)
    )
    
    layer31 = pooling_LeNetConvPoolLayer(
        rng,
        input=layer21_.output,
        filter_shape=(256, 128, 3, 3),
        poolsize=(2, 2)
    )
    layer32 = upsampling_LeNetConvPoolLayer(
        rng,
        input=layer31.output,
        filter_shape=(128, 256, 3, 3),
        poolsize=(2, 2),
        input_channels=256
    )   
    
    add_2=layer21_.output+layer32.output
    
    layer22 = upsampling_LeNetConvPoolLayer(
        rng,
        input=add_2,
        filter_shape=(64, 128, 3, 3),
        poolsize=(2, 2),
        input_channels=128
    )
    layer22_ = LeNetConvPoolLayer(
        rng,
        input=layer22.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
        poolsize=(1, 1)
    )
    
    add_1=layer11_.output+layer22_.output
    
    layer_output= LeNetConvPoolLayer(
        rng,
        input=add_1,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(3, 64, 3, 3),
        poolsize=(1, 1)
    )

    params = layer11.params + layer11_.params + layer_output.params + layer21.params + layer21_.params + layer22.params + layer22_.params + layer31.params+layer32.params
    square=layer_output.output.reshape((batch_size,3072))-y
    cost = T.mean(T.sum(T.pow(square,2)))
    grads = T.grad(cost, params)
    def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
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
     #create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer_output.output,
        givens={
            x: test_set_x[index: index+1]
        }
    )
    
    show_drop = theano.function(
        [index],
        drop_layer,
        givens={
            x: test_set_x[index: index+1]
        }
    )

    
    show_ground_truth = theano.function(
        [index],
        x,
        givens={
            x: test_set_x[index: index+1]
        }
    )
    
    validate_model = theano.function(
        [index],
        cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_model = theano.function(
        [index],
        cost,
        updates=Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    


    train_nn(show_ground_truth,show_drop, train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)

        
