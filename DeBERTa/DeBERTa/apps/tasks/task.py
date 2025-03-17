#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#

import os
import csv
import copy
from collections import OrderedDict,defaultdict,Counter
from collections.abc import Sequence
import numpy as np
from ...utils import get_logger
from ...utils import xtqdm as tqdm
from ...data import example_to_feature
from .metrics import *
import pickle
import torch
from ..models import SequenceClassificationModel
logger=get_logger()

__all__ = ['EvalData', 'Task']

def pad2d_withmask(tensor_list):
    # Determine the maximum size of the first dimension across all tensors
    max_size = max(t.shape[0] for t in tensor_list)
    if max_size < 20:
      max_size = 20 
      #print("./DeBERTa/apps/tasks hardcode max_size=10")
    num_tensors = len(tensor_list)
    feature_dim = tensor_list[0].shape[1]

    # Initialize the padded tensor with zeros
    padded_tensor = torch.zeros(num_tensors, max_size, feature_dim)

    # Initialize the mask with zeros (indicating padding)
    mask = torch.zeros(num_tensors, max_size, dtype=torch.bool)
    # Pad each tensor and update the mask
    for i, tensor in enumerate(tensor_list):
        padded_tensor[i, :tensor.shape[0],:tensor.shape[1]] = tensor
        mask[i, :tensor.shape[0]] = 1

    return padded_tensor, mask


class EvalData:
  def __init__(self, name, examples, metrics_fn=None, predict_fn=None, ignore_metric=False, critial_metrics=None):
    def accuracy_fn(logits, labels):
      return OrderedDict(accuracy= metric_accuracy(logits, labels))

    def default_pred_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=-1)
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,p in enumerate(preds):
          fs.write('{}\t{}\n'.format(i, p))
    self.name = name
    self.data = examples
    self.ignore_metric = ignore_metric
    self.critial_metrics = critial_metrics
    self.metrics_fn = metrics_fn if metrics_fn is not None else accuracy_fn
    self.predict_fn = predict_fn if predict_fn is not None else default_pred_fn

  def __repr__(self):
    return f'{self.name}, {type(self.data)}: {len(self.data)}, {self.predict_fn}, {self.metrics_fn}'

class Task():
  _meta={}

  def __init__(self, tokenizer, args, **kwargs):
    self.tokenizer = tokenizer
    self.args = args
    self.cluster_path = args.cluster_path

    if self.cluster_path is not None:
      print("Load cluster",self.cluster_path)
      with open(self.cluster_path, 'rb') as handle:
          self.token_emb_dict = pickle.load(handle)
  
  
  def eval_data(self, **kwargs):
    raise NotImplementedError('Eval_data method not implemented yet.')

  def train_data(self, **kwargs):
    raise NotImplementedError('Eval_data method not implemented yet.')

  def test_data(self, **kwargs):
    raise NotImplementedError('Eval_data method not implemented yet.')

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  def label2id(self, labelstr):
    label_dict = {l:i for i,l in enumerate(self.get_labels())}
    return label_dict[labelstr] if labelstr in label_dict else -1

  def get_train_fn(self, *args, **kwargs):
    return None

  def get_eval_fn(self, *args, **kwargs):
    return None

  def get_pred_fn(self, *args, **kwargs):
    return None

  def get_loss_fn(self, *args, **kwargs):
    return None

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      return OrderedDict(accuracy= metric_accuracy(logits, labels))
    return metrics_fn

  def get_predict_fn(self):
    """Calcuate metrics based on prediction results"""
    def predict_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=-1)
      labels = self.get_labels()
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,p in enumerate(preds):
          fs.write('{}\t{}\n'.format(i, labels[p]))

    return predict_fn

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None, label_type='int', training=False):
    tokenizer = self.tokenizer

    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      data =  example_to_feature(tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, label_type=label_type, **kwargs)


      token_ids = data['input_ids']

      if self.cluster_path is not None:
        cluster_mean_list = []
        for i in range(token_ids.shape[0]):
            element = token_ids[i].item()
            if  element not in  self.token_emb_dict or self.token_emb_dict[element] is None  or len(self.token_emb_dict[element]) == 0  :
                cluster_mean_list.append(torch.zeros(1,1024).half())
            else:
                '''
                if torch.equal(self.token_emb_dict[element] , torch.tensor([0])) :
                  print("[zero]", element)
                  cluster_mean_list.append(torch.zeros(1,1024).half())
                  print(example)
                  print(token_ids)
                else:
                '''
                cluster_mean_list.append(self.token_emb_dict[element])
  

        cluster_mean, cluster_padding_filter = pad2d_withmask(cluster_mean_list)
        data['cluster_mean'] = cluster_mean

        if self.args.fp16:
          data['cluster_mean'] = data['cluster_mean'].half()
        data['cluster_padding_filter'] = cluster_padding_filter
  
      return data
    return _example_to_feature

  def get_model_class_fn(self):
    return SequenceClassificationModel.load_model
  
  @classmethod
  def add_arguments(cls, parser):
    """Add task specific arguments
      e.g. parser.add_argument('--data_dir', type=str, help='The path of data directory.')
    """
    parser.add_argument('--task_example_arg', type=str, default=None, help='An example task specific argument')

    return parser
