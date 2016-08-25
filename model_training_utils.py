import _init_paths
import os
import numpy as np
import h5py
import time
import datetime
import sg_utils
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import traceback as tb
import code
try:
    from python_layers.loss_tracking_layer import loss_tracker_dict;
    # global loss_tracker_dict;
except:
    print 'FAILED at loss_tracker_dict'
    time.sleep(1)

__author__ = "Ishan Misra <ishanmisra@gmail.com>"
__date__ = "2016.07.24"

#Utilities to train models and log different aspects
#Based on train_net.py from Ross Girshick's Fast-RCNN codebase

class WatchTrainer():
    """ A class to watch training and keep track of a bunch of things like activations, weight norms, diffs """
    def __init__(self, solverPath, solver=None, checkSolver=True, verbose=True):
        assert( os.path.isfile(solverPath) ), 'solver: %s does not exist'%(solverPath);
        assert( solver is not None), 'none solver is not implemented yet';
        #TODO: add none solver option, if solver is none then init solver using caffe
        self.solverPath = solverPath;
        self.parse_solver();
        self.solver = solver;
        if checkSolver:
            self.check_solver();
        self.logNames = {};
        self.isLogging = False;
        self.prevWts = None;
        self.verbose = verbose;

    def parse_solver(self):
        solverPath = self.solverPath;
        self.expName = os.path.split(solverPath)[-1].split('_')[0];
        self.expDir = os.path.split(solverPath)[0];
        self.solver_param = caffe_pb2.SolverParameter();
        with open(self.solverPath, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        allLines = [x.strip() for x in open(solverPath,'r')];

        snapPath = self.solver_param.snapshot_prefix;
        snapExp = os.path.split(snapPath)[-1];
        snapPath = os.path.split(snapPath)[0];
        sg_utils.mkdir(snapPath);
        assert( os.path.isdir(snapPath) ), '%s does not exist'%(snapPath);
        self.snapPath = snapPath;
        assert( self.snapPath == os.path.split(self.solver_param.snapshot_prefix)[0] );

    def check_solver(self):
        #assumes solver has the following first 2 lines
        #train_net: "blah"
        #snapshot: "blah"
        #check if solver points to the correct train proto
        solverPath = self.solverPath;
        expName = os.path.split(solverPath)[-1].split('_')[0];

        allLines = [x.strip() for x in open(solverPath,'r')];
        trainNet = allLines[0].split(':')[1].strip();
        trainNet = os.path.split(trainNet)[-1];
        trainExp = trainNet.split('_')[0].replace('"','');
        assert( expName == trainExp ), 'train proto: %s %s'%(expName, trainExp);

        snapPath = self.solver_param.snapshot_prefix;
        snapExp = os.path.split(snapPath)[-1];
        snapExp = snapExp.split('_')[0];
        assert( expName == snapExp ), 'snapshot name: %s %s'%(expName, snapExp);
        print 'solver paths seem correct'
        print 'will snap to ', snapPath

    def get_time_str(self):
        ts = time.time();
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        return st;

    def init_logging(self):
        if bool(self.logNames):
            #dict is not empty, so we already have lognames set
            return;
        st = self.get_time_str();
        self.logNames['weight_norms'] = os.path.join(self.expDir,\
                         'logs', self.expName + '_' + st + '_weight-norm_.h5')
        self.logNames['weight_activations'] = os.path.join(self.expDir,\
                         'logs', self.expName + '_' + st +  '_weight-activations_.h5')
        self.logNames['weight_diffs'] = os.path.join(self.expDir,\
                         'logs', self.expName  + '_' + st + '_weight-diffs_.h5')
        self.logNames['weight_meta'] = os.path.join(self.expDir,\
                         'logs', self.expName + '_weight-meta_' + '.pkl')
        self.logNames['loss_tracker'] = os.path.join(self.expDir,\
                         'logs', self.expName + '_' + st + '_loss-tracker' + '.pkl')
        self.isLogging = True;

    def model_weight_activations(self):
        net = self.solver.net;
        layers = net.blobs.keys();
        currActivMeans=[];
        for layer in layers:
            meanActiv = net.blobs[layer].data.mean();
            currActivMeans.append(meanActiv);
        currActivMeans=np.array(currActivMeans);
        if os.path.isfile(self.logNames['weight_activations']):
            ss = sg_utils.load(self.logNames['weight_activations']);
            currActivMeans = np.dstack( (ss['activmeans'], currActivMeans) );
        fh = h5py.File(self.logNames['weight_activations'],'w');
        fh.create_dataset('activmeans',data=currActivMeans,dtype=np.float32);
        fh.close();

    def model_weight_diffs(self):
        net = self.solver.net;
        layers = net.params.keys(); #params is an ordered dict, so keys are ordered
        currwtdiffs = [];
        for layer in layers:
            numObj = len(net.params[layer]);
            means = {};
            medians = {};
            for b in range(numObj):
                wtdiff = net.params[layer][b].diff;
                currwtdiffs.append(np.linalg.norm(wtdiff));
        currwtdiffs = np.array(currwtdiffs);
        if os.path.isfile(self.logNames['weight_diffs']):
            ss = sg_utils.load(self.logNames['weight_diffs']);
            currwtdiffs = np.dstack( (ss['wtdiffs'], currwtdiffs) );
        fh = h5py.File(self.logNames['weight_diffs'],'w')
        fh.create_dataset('wtdiffs',data=currwtdiffs,dtype=np.float32);
        fh.close();

    def model_weight_stats(self):
        net = self.solver.net;
        outFile = self.logNames['weight_norms'];
        prevWts = self.prevWts;

        layers = net.params.keys(); #params is an ordered dict, so keys are ordered
        currmeans = [];
        currnorms = [];
        currwtdiffnorm = [];
        for layer in layers:
            numObj = len(net.params[layer]);
            means = {};
            medians = {};
            for b in range(numObj):
                wtsshape = net.params[layer][b].data.shape;
                wts = net.params[layer][b].data.astype(np.float32, copy=False);
                if prevWts is not None:
                    wtdiff = prevWts[layer][b] - wts;
                    wtdiffnorm = np.linalg.norm(wtdiff);
                    currwtdiffnorm.append(wtdiffnorm);
                wtsmean = wts.mean();
                wtsnorm = np.linalg.norm(wts);
                currmeans.append(wtsmean);
                currnorms.append(wtsnorm);
        currmeans=np.array(currmeans);
        currnorms=np.array(currnorms);
        currwtdiffnorm=np.array(currwtdiffnorm);
        if os.path.isfile(outFile):
            ss = sg_utils.load(outFile);
            means = ss['means'];
            norms = ss['norms'];
            means = np.dstack((means, currmeans));
            norms = np.dstack((norms, currnorms));
            if prevWts is not None:
                if 'wtdiffnorms' in ss:
                    wtdiffnorms = ss['wtdiffnorms'];
                    wtdiffnorms = np.dstack((wtdiffnorms, currwtdiffnorm));
                else:
                    wtdiffnorms = currwtdiffnorm;
        else:
            means=currmeans;
            norms=currnorms;
            wtdiffnorms=currwtdiffnorm;
        if self.verbose:
            print '%d writing to weight_norms ... '%(self.solver.iter),
        try:
            fh = h5py.File(outFile,'w'); #overwrite!!
            fh.create_dataset('means',data=means,dtype=np.float32)
            fh.create_dataset('norms',data=norms,dtype=np.float32)
            if prevWts is not None:
                fh.create_dataset('wtdiffnorms',data=wtdiffnorms,dtype=np.float32)
            fh.close();
        except:
            tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
            try:
                fh.close();
                print 'error when writing to log'
            except:
                print 'error when writing to log'
            pass;
        if self.verbose:
            print 'success';

    def model_track_loss(self):
        if not self.track_indiv_loss:
            return;
        from python_layers.loss_tracking_layer import loss_tracker_dict;
        indiv_losses = np.array(loss_tracker_dict['indiv_losses'])
        indiv_labs = np.array(loss_tracker_dict['indiv_labs']).astype(np.int32)
        indiv_preds = np.array(loss_tracker_dict['indiv_preds']).astype(np.int32)
        indiv_probs = np.array(loss_tracker_dict['indiv_probs']).astype(np.float32)
        if os.path.exists(self.logNames['loss_tracker']):
            dt = sg_utils.load(self.logNames['loss_tracker']);
            indiv_losses = np.concatenate((dt['indiv_losses'], indiv_losses));
            indiv_labs = np.concatenate((dt['indiv_labs'], indiv_labs));
            indiv_preds = np.concatenate((dt['indiv_preds'], indiv_preds));
            indiv_probs = np.concatenate((dt['indiv_probs'], indiv_probs));
        try:
            sg_utils.save(self.logNames['loss_tracker'], [indiv_losses, indiv_labs, indiv_preds, indiv_probs],\
                                ['indiv_losses', 'indiv_labs', 'indiv_preds', 'indiv_probs'], overwrite=True)
            loss_tracker_dict['indiv_losses'] = [];
            loss_tracker_dict['indiv_labs'] = [];
            loss_tracker_dict['indiv_probs'] = [];
            loss_tracker_dict['indiv_preds'] = [];
            print 'saved losses'
        except:
            print 'error with loss tracker'



    def get_model_weights(self):
        net = self.solver.net;
        srcWeights = {};
        for layer in net.params:
            srcWeights[layer] = [];
            for b in range(len(net.params[layer])):
                srcWeights[layer].append( net.params[layer][b].data.astype(dtype=np.float32, copy=True));
        return srcWeights;

    def snapshot(self):
        net = self.solver.net

        if not os.path.exists(self.snapPath):
            sg_utils.mkdir(self.snapPath);

        filename = self.expName + '_snapshot_' + 'iter_{:d}'.format(self.offset_iter + self.solver.iter) + '.caffemodel';
        filename = os.path.join(self.snapPath, filename);
        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)
        return filename;

    def train_model(self, max_iters, log_iter, snapshot_iter, track_indiv_loss=False, offset_iter=0):
        last_snapshot_iter = -1;
        self.offset_iter = offset_iter;
        assert snapshot_iter % log_iter == 0, 'logging and snapshotting must be multiples';
        if self.isLogging:
            layers = self.solver.net.params.keys(); #params is an ordered dict, so keys are ordered
            layer_param_shapes = {};
            for layer in self.solver.net.params:
                layer_param_shapes[layer] = [];
                for b in range(len(self.solver.net.params[layer])):
                    layer_param_shapes[layer].append(self.solver.net.params[layer][b].data.shape)
            sg_utils.save(self.logNames['weight_meta'], [layers, layer_param_shapes], ['layer_names', 'layer_param_shapes'], overwrite=True);

        #setup losstracker
        if track_indiv_loss:
            self.track_indiv_loss = track_indiv_loss;
            check_loss_tracker = True;
        else:
            check_loss_tracker = False;
        #try snapshotting
        tmp = self.offset_iter;
        self.offset_iter = -1;
        print 'trying snapshot'
        filename = self.snapshot();
        # os.remove(filename);
        self.offset_iter = tmp;

        print 'snapshotting worked: %s'%(filename);
        while self.solver.iter < max_iters:
            if self.isLogging and \
                (self.solver.iter % log_iter == 0 or self.solver.iter == 0):
                self.model_weight_stats()
                self.model_weight_activations();
                self.model_weight_diffs();
                self.prevWts = self.get_model_weights();
            self.solver.step(log_iter)
            if self.solver.iter % snapshot_iter == 0 or check_loss_tracker:
                last_snapshot_iter = self.solver.iter
                self.snapshot()
                self.model_track_loss()
                check_loss_tracker = False;
        if last_snapshot_iter != self.solver.iter:
            self.snapshot()
