import caffe
import numpy as np
import yaml
import h5py
import math
import code
import traceback as tb
import os
import time
import sys
import operator
import matplotlib.pyplot as plt
import _init_paths

#Layer that keeps a track of losses for each datapoint
#Helps visualize if our network is doing something meaningful, or just "converging to the mean"
global loss_tracker_dict;
loss_tracker_dict = {};

class LossTrackingLayer(caffe.Layer):

    def softmax(self, softmax_inputs, temp=1.0):
        softmax_inputs = softmax_inputs.astype(np.float64)
        tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
        shifted_inputs = softmax_inputs - softmax_inputs.max(axis=0)
        exp_outputs = np.exp(temp * shifted_inputs)
        exp_outputs_sum = exp_outputs.sum()
        if np.isnan(exp_outputs_sum):
            return exp_outputs * float('nan')
        assert exp_outputs_sum > 0
        if np.isinf(exp_outputs_sum):
            return np.zeros_like(exp_outputs)
        eps_sum = 1e-20
        return exp_outputs / max(exp_outputs_sum, eps_sum)

    def setup(self, bottom, top):
        """Setup the TripleTupleSamplingLayer."""
        layer_params = yaml.load(self.param_str); #new version of caffe
        assert len(top) == 1
        #two bottoms
        assert len(bottom) == 2
        #bottom[0] is prob, bottom[1] is label
        assert bottom[0].shape[0] == bottom[1].shape[0], '{} {}'.format(tuple(bottom[0].shape), tuple(bottom[1].shape))
        assert bottom[0].shape[1] >= bottom[1].shape[1]
        if len(bottom[0].shape) >= 3:
            bottom[0].shape[2] == 1
        if len(bottom[0].shape) >= 4:
            bottom[0].shape[3] == 1
        assert bottom[1].shape[1] == 1 and bottom[1].shape[2] == 1 and bottom[1].shape[3] == 1

        self._flt_min = 1e-12;
        global loss_tracker_dict;
        loss_tracker_dict = {};
        loss_tracker_dict['indiv_losses'] = [];
        loss_tracker_dict['indiv_probs'] = [];
        loss_tracker_dict['indiv_labs'] = [];
        loss_tracker_dict['indiv_preds'] = [];
        top_shape = tuple([1,1,1,1])
        top[0].reshape(*top_shape)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        global loss_tracker_dict;
        prob = bottom[0].data;
        labs = bottom[1].data.astype(np.int32);
        for ii in range(bottom[0].shape[0]):
            loss_tracker_dict['indiv_losses'].append( -np.log(np.maximum(self._flt_min, prob[ii,labs[ii]].flatten()))[0] );
        loss_tracker_dict['indiv_labs'].extend(labs.flatten());
        loss_tracker_dict['indiv_preds'].extend( np.argmin(prob,axis=1).flatten());
        loss_tracker_dict['indiv_probs'].extend( np.squeeze(prob).astype(np.float32) );
        top[0].data[0] = sum(loss_tracker_dict['indiv_losses'])

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
