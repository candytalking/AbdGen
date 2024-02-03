"""
This file is a copyrighted under the BSD 3-clause licence, details of which can be found in the root directory.
Code for
Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings
https://arxiv.org/abs/2310.17451

"""

import os
import numpy as np
import torch
import random
import glob


class PathManager:
    def __init__(self, path_object, dataset, exp_name):
        self.exp_root_path = path_object['exp_root_path']
        self.dataset_path = path_object['dataset_path']
        self.model_path = path_object['model_path']
        self.tmp_path = path_object['tmp_path']
        self.rule_path = path_object['rule_path']
        self.result_path = path_object['result_path']
        self.pl_tmp_path = path_object['pl_tmp_path']
        self.dataset = dataset
        self.exp_name = exp_name

    def get_spec_path(self, path_type):
        if path_type == 'model':
            path = os.path.join(self.exp_root_path, self.model_path, self.dataset, self.exp_name)
        elif path_type == 'pl':
            path = os.path.join(self.exp_root_path, self.pl_tmp_path, self.dataset, self.exp_name)
        elif path_type == 'result':
            path = os.path.join(self.exp_root_path, self.result_path, self.dataset, self.exp_name)
        else:
            raise NameError('unsupported mode.')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_spec_file(self, path_type, file_name=None):
        if path_type == 'model':
            return os.path.join(self.exp_root_path, self.model_path, self.dataset, self.exp_name, file_name)
        elif path_type == 'pl':
            return os.path.join(self.exp_root_path, self.pl_tmp_path, self.dataset, self.exp_name, file_name)

    def get_gen_file(self, path_type, file_name):
        if path_type == 'dataset':
            return os.path.join(self.exp_root_path, self.dataset_path, file_name)


def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
  }, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path):
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  return epoch, loss


def find_latest_checkpoint(checkpoint_dir):
  checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
  if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint
  else:
    return None










