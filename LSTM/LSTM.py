
from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import copy
import numpy
import os
import random
import timeit

import theano
from theano import tensor as T

from hw4_utils import load_data, contextwin, shuffle, conlleval, check_dir

# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(1500)

class LSTM(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, nh, nh2, nc, ne, de, cs, normal=True):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the first hidden layer
        
        :type nh2: int
        :param nh: dimension of the second hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        self.emb = theano.shared(name='emb',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (ne+1, de))
                                .astype(theano.config.floatX)) 
        
        self.wxi = theano.shared(name='wxi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de*cs, nh))
                                .astype(theano.config.floatX))
        self.wxf = theano.shared(name='wxf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de*cs, nh))
                                .astype(theano.config.floatX))
        self.wxc = theano.shared(name='wxc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de*cs, nh))
                                .astype(theano.config.floatX)) 
        self.wxo = theano.shared(name='wxo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de*cs, nh))
                                .astype(theano.config.floatX))        
        
        self.whi = theano.shared(name='whi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.whf = theano.shared(name='whf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))        
        self.whc = theano.shared(name='whc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.who = theano.shared(name='who',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        
        self.wci = theano.shared(name='wci',value=0.2 * numpy.random.uniform(-1.0,1.0,nh).astype(theano.config.floatX))
        self.wcf = theano.shared(name='wcf',value=0.2 * numpy.random.uniform(-1.0,1.0,nh).astype(theano.config.floatX))
        self.wco = theano.shared(name='wco',value=0.2 * numpy.random.uniform(-1.0,1.0,nh).astype(theano.config.floatX))
        
        
        
        self.bi = theano.shared(name='bi',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bc = theano.shared(name='bc',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bf = theano.shared(name='bf',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo = theano.shared(name='bo',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.C0 = theano.shared(name='C0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc)))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))                       

        # bundle

        
       
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence')  # labels


        def recurrence(x_t, h_tm1, C_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wxi) + T.dot(h_tm1, self.whi) + T.dot(C_tm1, T.nlinalg.diag(self.wci)) + self.bi)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wxf) + T.dot(h_tm1, self.whf) + T.dot(h_tm1, T.nlinalg.diag(self.wcf)) + self.bf)
            C_t = f_t * C_tm1 + i_t * T.tanh(T.dot(x_t,self.wxc) + T.dot(h_tm1,self.whc) + self.bc)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wxo) + T.dot(h_tm1, self.who) + T.dot(C_t, T.nlinalg.diag(self.wco)) + self.bo)
            h_t = o_t * T.tanh(C_t)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t,C_t,s_t]

        # bundle
        self.params = [self.emb,self.wxi, self.wxf, self.wxc, self.wxo, self.whi,self.whf,self.whc,self.who,
                       self.wci,self.wcf,self.wco,
                       self.bi,self.bc,self.bf,self.bo,self.w,self.b]

        #[h, h2], _ = theano.scan(fn=recurrence,
        #                        sequences=x,
        #                        outputs_info=[self.h0,self.h02],
        #                        n_steps=x.shape[0])
        [h,c,s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0,self.C0,None],
                                n_steps=30)

        
        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(30), y_sentence[0:30]])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates,
                                              allow_input_downcast=True
                                             )
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb /
                                                  T.sqrt((self.emb**2)
                                                  .sum(axis=1))
                                                  .dimshuffle(0, 'x')})
        self.normal = normal

    def train(self, x, y, window_size, learning_rate):

        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        self.sentence_train(words, labels, learning_rate)
        if self.normal:
            self.normalize()

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))

def test_lstm(**kwargs):

    # process input arguments
    param = {
        'fold': 3,
        'lr': 0.1,
        'verbose': True,
        'decay': True,
        'win': 3,
        'nhidden1': 300,
        'nhidden2': 0,
        'seed': 345,
        'emb_dimension': 50,
        'nepochs': 20,
        'savemodel': False,
        'folder':'../result'}
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

    # create result folder if not exists
    check_dir(param['folder'])

    # load the dataset
    print('... loading the dataset')
    train_set, valid_set, test_set, dic = load_data(param['fold'])

    # create mapping from index to label, and index to word
    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    
    train_lex_temp, train_ne, train_y_temp = train_set
    valid_lex_temp, valid_ne, valid_y_temp = valid_set
    test_lex_temp, test_ne, test_y_temp = test_set
    
  
    train_lex = (7) * numpy.ones((len(train_lex_temp),30))
    for i in range(len(train_lex_temp)):
        for j in range(min(30,len(train_lex_temp[i]))):
            train_lex[i][j] = train_lex_temp[i][j]
            
    test_lex = (7) * numpy.ones((len(test_lex_temp),30))
    for i in range(len(test_lex_temp)):
        for j in range(min(30,len(test_lex_temp[i]))):
            test_lex[i][j] = test_lex_temp[i][j]
            
    valid_lex = (7) * numpy.ones((len(valid_lex_temp),30))
    for i in range(len(valid_lex_temp)):
        for j in range(min(30,len(valid_lex_temp[i]))):
            valid_lex[i][j] = valid_lex_temp[i][j]
            
    train_y = 126 * numpy.ones((len(train_y_temp),30))
    for i in range(len(train_y_temp)):
        for j in range(min(30,len(train_y_temp[i]))):
            train_y[i][j] = train_y_temp[i][j]
            
    test_y = 126 * numpy.ones((len(test_y_temp),30))
    for i in range(len(test_y_temp)):
        for j in range(min(30,len(test_y_temp[i]))):
            test_y[i][j] = test_y_temp[i][j]
            
    valid_y = 126 * numpy.ones((len(valid_y_temp),30))
    for i in range(len(valid_y_temp)):
        for j in range(min(30,len(valid_y_temp[i]))):
            valid_y[i][j] = valid_y_temp[i][j]

    


    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
    words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]
    groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
    words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    # TODO
    print('... building the model')
    rnn = LSTM(
        nh=param['nhidden1'],
        nh2=param['nhidden2'],
        nc=nclasses,
        ne=vocsize,
        de=param['emb_dimension'],
        cs=param['win'])

    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):

        # shuffle
        shuffle([train_lex, train_ne, train_y], param['seed'])

        param['ce'] = e
        tic = timeit.default_timer()
        ##########################
        #print(rnn.showh2(numpy.asarray(contextwin(test_lex[0], param['win'])).astype('int32')))
        #########################
        for i, (x, y) in enumerate(zip(train_lex, train_y)):
            rnn.train(x, y, param['win'], param['clr'])

            print('[learning] epoch %i >> %2.2f%%' % (
                e, (i + 1) * 100. / nsentences), end=' ')
            print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic), end='')
            sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        predictions_test = [map(lambda x: idx2label[x],
                            rnn.classify(numpy.asarray(
                            contextwin(x, param['win'])).astype('int32')))
                            for x in test_lex]
        predictions_valid = [map(lambda x: idx2label[x],
                             rnn.classify(numpy.asarray(
                             contextwin(x, param['win'])).astype('int32')))
                             for x in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test,
                             groundtruth_test,
                             words_test,
                             param['folder'] + '/current.test.txt',
                             param['folder'])
        res_valid = conlleval(predictions_valid,
                              groundtruth_valid,
                              words_valid,
                              param['folder'] + '/current.valid.txt',
                              param['folder'])

        if res_valid['f1'] > best_f1:

            if param['savemodel']:
                rnn.save(param['folder'])

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid['f1']

            if param['verbose']:
                print('NEW BEST: epoch', e,
                      'valid F1', res_valid['f1'],
                      'best test F1', res_test['f1'])

            param['vf1'], param['tf1'] = res_valid['f1'], res_test['f1']
            param['vp'], param['tp'] = res_valid['p'], res_test['p']
            param['vr'], param['tr'] = res_valid['r'], res_test['r']
            param['be'] = e

            os.rename(param['folder'] + '/current.test.txt',
                      param['folder'] + '/best.test.txt')
            os.rename(param['folder'] + '/current.valid.txt',
                      param['folder'] + '/best.valid.txt')
        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
           'valid F1', param['vf1'],
           'best test F1', param['tf1'],
           'with the model', param['folder'])

if __name__ == '__main__':
    test_lstm()
