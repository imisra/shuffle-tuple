import numpy as np
import cPickle
import heapq
import os
from IPython.core.debugger import Tracer
import scipy.io as scio
import time
import h5py
import json

def get_file_list(dirPath, extension = None):
    onlyfiles = [ os.path.join(dirPath,f) for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath,f)) ];
    if extension!= None:
        onlyfiles = [f for f in onlyfiles if os.path.splitext(f)[1]==extension];
    onlyfiles.sort();
    return onlyfiles;

def list_to_indexed_dict(lvar):
    dvar = {};
    for ind, item in enumerate(lvar):
      dvar[item]=ind;
    return dvar;

def tic_toc_print(interval, string):
  global tic_toc_print_time_old
  if 'tic_toc_print_time_old' not in globals():
    tic_toc_print_time_old = time.time()
    print string
  else:
    new_time = time.time()
    if new_time - tic_toc_print_time_old > interval:
      tic_toc_print_time_old = new_time;
      print string
def mkdir(output_dir):
    return mkdir_if_missing(output_dir);

def mkdir_if_missing(output_dir):
  """
  def mkdir_if_missing(output_dir)
  """
  if not os.path.exists(output_dir):
    try:
      os.makedirs(output_dir)
      return True;
    except: #generally happens when many processes try to make this dir
      return False;

def save_variables_h5(h5_file_name, var, info, overwrite = False):
  if info is None:
    return save_variables_h5_dict(h5_file_name, var, overwrite)
  if os.path.exists(h5_file_name) and overwrite == False:
    raise Exception('{:s} exists and over write is false.'.format(h5_file_name))
  # Construct the dictionary
  assert(type(var) == list); assert(type(info) == list);
  with h5py.File(h5_file_name, 'w') as f:
    for i in range(len(info)):
      d = f.create_dataset(info[i],data=var[i]);

def rec_get_keys(fh, src, keyList):
    if src!='' and type(fh[src]).__name__ == 'Dataset':
      keyList.append(src);
      return keyList;
    if src!='':
      moreSrcs = fh[src].keys();
    else:
      moreSrcs = fh.keys();
    for kk in moreSrcs:
      if src=='':
          keyList = rec_get_keys(fh, kk, keyList);
      else:
          keyList = rec_get_keys(fh, src+'/'+kk, keyList);
    return keyList;

def get_h5_keys(h5_file_name):
  if os.path.exists(h5_file_name):
    with h5py.File(h5_file_name,'r') as f:
      keyList = rec_get_keys(f, '', []);
    return keyList;
  else:
    raise Exception('{:s} does not exists.'.format(h5_file_name))


def save_variables_h5_dict(h5_file_name, dictVar, overwrite = False):
  if os.path.exists(h5_file_name) and overwrite == False:
    raise Exception('{:s} exists and over write is false.'.format(h5_file_name))
  # Construct the dictionary
  assert(type(dictVar) == dict);
  with h5py.File(h5_file_name, 'w') as f:
    for key in dictVar:
      d = f.create_dataset(key, data=dictVar[key], compression="gzip", compression_opts=9);

def load_variablesh5(h5_file_name):
  if os.path.exists(h5_file_name):
    with h5py.File(h5_file_name,'r') as f:
      d = {};
      h5keys = get_h5_keys(h5_file_name);
      for key in h5keys:
        d[key] = f[key].value;
    return d
  else:
    raise Exception('{:s} does not exists.'.format(h5_file_name))

def save_variables(pickle_file_name, var, info, overwrite = False):
  """
    def save_variables(pickle_file_name, var, info, overwrite = False)
  """
  fext = os.path.splitext(pickle_file_name)[1]
  if fext =='.h5':
    return save_variables_h5(pickle_file_name, var, info, overwrite);

  elif fext == '.pkl':  
    if os.path.exists(pickle_file_name) and overwrite == False:
      raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
    if info is not None:
      # Construct the dictionary
      assert(type(var) == list); assert(type(info) == list);
      d = {}
      for i in xrange(len(var)):
        d[info[i]] = var[i]
    else: #we have the dictionary in var
      d = var;
    with open(pickle_file_name, 'wb') as f:
      cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)
  else:
    raise Exception('{:s}: extension unknown'.format(fext))

def load_variables(pickle_file_name):
  """
  d = load_variables(pickle_file_name)
  Output:
    d     is a dictionary of variables stored in the pickle file.
  """
  fext = os.path.splitext(pickle_file_name)[1]
  if fext =='.h5':
    return load_variablesh5(pickle_file_name);

  elif fext == '.pkl':
    if os.path.exists(pickle_file_name):
      with open(pickle_file_name, 'rb') as f:
        d = cPickle.load(f)
      return d
    else:
      raise Exception('{:s} does not exists.'.format(pickle_file_name))
  elif fext == '.json':
    with open(pickle_file_name, 'r') as fh:
        data = json.load(fh)
    return data
  else:
    raise Exception('{:s}: extension unknown'.format(fext))

#wrappers for load_variables and save_variables
def load(pickle_file_name):
    return load_variables(pickle_file_name);

def save(pickle_file_name, var, info, overwrite = False):
    return save_variables(pickle_file_name, var, info, overwrite);

