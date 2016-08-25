import _init_paths
import caffe
import numpy as np
import code
import traceback as tb
import os
from model_training_utils import WatchTrainer
from caffe import layers as L

caffe.set_device(1)
caffe.set_mode_gpu()

solverPath = 'tuple_solver.prototxt'
solver = caffe.SGDSolver(solverPath)

numIter = 100000;
logStep = 20;
snapshotIter = 20000;
trainer = WatchTrainer(solverPath, solver);
trainer.init_logging();
trainer.train_model(numIter, logStep, snapshotIter, track_indiv_loss=True);
