"""
Author: Ishan Misra
Email: ishan@cmu.edu

Modification of the data layer used by https://github.com/mitmul/deeppose for Caffe
"""
import _init_paths
import caffe
import numpy as np
import yaml
import six
from multiprocessing import Process, Queue, Array
import multiprocessing
import h5py
import math
import code
import traceback as tb
import os
from PIL import Image
import cv2
import scipy.misc
from multiprocessing.sharedctypes import Array as sharedArray
import ctypes
import atexit
import time
import sys
import operator
import matplotlib.pyplot as plt
import timer
from transform_dictishan import Transform

def prod(ll):
    return float(reduce(operator.mul, ll, 1));

def load_data(args, input_q, minibatch_q):
    c = args['channel']
    s = args['size']
    d = args['joint_num'] * 2

    input_data_base = Array(ctypes.c_float, args['batchsize'] * c * s * s)
    input_data = np.ctypeslib.as_array(input_data_base.get_obj())
    input_data = input_data.reshape((args['batchsize'], c, s, s))

    label_base = Array(ctypes.c_float, args['batchsize'] * d)
    label = np.ctypeslib.as_array(label_base.get_obj())
    label = label.reshape((args['batchsize'], d))

    x_queue, o_queue = Queue(), Queue()
    workers = [Process(target=transform,
                       args=(args, x_queue, args['datadir'], args['fname_index'],
                             args['joint_index'], o_queue))
               for _ in range(1)]
    for w in workers:
        w.start()

    while True:
        x_batch = input_q.get()
        if x_batch is None:
            break

        # data augmentation
        for x in x_batch:
            x_queue.put(x)
        j = 0
        while j != len(x_batch):
            a, b = o_queue.get()
            input_data[j] = a
            label[j] = b
            j += 1
        minibatch_q.put([input_data, label])

    for _ in range(self.batchsize):
        x_queue.put(None)
    for w in workers:
        w.join()

def transform(self, x_queue, datadir, fname_index, joint_index, o_queue):
    trans = Transform(self)
    while True:
        x = x_queue.get()
        if x is None:
            break
        x, t = trans.transform(x.split(','), datadir, fname_index, joint_index)
        o_queue.put((x.transpose((2, 0, 1)), t))



class PoseDataLayer(caffe.Layer):
    def load_dataset(self):
        train_fn = '%s/train_joints.csv' % self.datadir
        test_fn = '%s/test_joints.csv' % self.datadir
        train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
        test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

        return train_dl, test_dl

    def setup(self, bottom, top):
        """Setup the KeyHDF5Layer."""
        layer_params = yaml.load(self.param_str); #new version of caffe
        self.datadir = layer_params['datadir']
        self.train_dl, self.test_dl = self.load_dataset()
        self.N, self.N_test = len(self.train_dl), len(self.test_dl)
        self.channel = 3;
        self.size=layer_params['size'];
        self.crop_pad_inf=1.5;
        self.crop_pad_sup=2.0;
        self.shift=5;
        self.lcn=1;
        self.batchsize=32;
        if 'seed' in layer_params:
            self.seed = layer_params['seed']
        else:
            self.seed=0;
        if 'num_labels' in layer_params:
            self.num_labels = layer_params['num_labels']
            self.joint_num=self.num_labels/2;
        else:
            self.num_labels=14;
            self.joint_num=7;

        arg_dict = {};
        arg_dict['channel']=self.channel
        arg_dict['size']=self.size
        arg_dict['crop_pad_inf']=self.crop_pad_inf
        arg_dict['crop_pad_sup']=self.crop_pad_sup
        arg_dict['lcn']=self.lcn
        arg_dict['joint_num']=self.joint_num
        arg_dict['batchsize']=self.batchsize
        arg_dict['joint_index']=1;
        arg_dict['fname_index']=0;
        arg_dict['datadir']=self.datadir
        arg_dict['cropping']=1;
        arg_dict['flip']=1;
        arg_dict['shift']=5;

        self._arg_dict = arg_dict;
        self.input_q, self.minibatch_q = Queue(), Queue(maxsize=1)
        data_loader = Process(target=load_data,
                          args=(arg_dict, self.input_q, self.minibatch_q))
        data_loader.start()

        N = len(self.train_dl);
        self.numepochs=2000;
        np.random.seed(self.seed)

        for n in range(self.numepochs):
            perm = np.random.permutation(N)
            for i in six.moves.range(0, N, arg_dict['batchsize']):
                self.input_q.put(self.train_dl[perm[i:i + arg_dict['batchsize']]])

        #do the tops make sense
        assert (len(top) == 2)
        self.data_shapes = [ [self.batchsize,3,self.size,self.size], [self.batchsize,self.num_labels,1,1] ]
        top[0].reshape(*(tuple(self.data_shapes[0])))
        top[1].reshape(*(tuple(self.data_shapes[1])))

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        input_data, label = self.minibatch_q.get()
        top[0].data[...]=input_data.astype(np.float32,copy=True)
        top[1].data[:,:,0,0]=label.astype(np.float32,copy=True);


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        top[0].reshape(*(tuple(self.data_shapes[0])))
        top[1].reshape(*(tuple(self.data_shapes[1])))
