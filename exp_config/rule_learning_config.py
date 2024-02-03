"""
This file is a copyrighted under the BSD 3-clause licence, details of which can be found in the root directory.
Code for
Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings
https://arxiv.org/abs/2310.17451

"""

import torch
from data_management.dataloader import *


mario_config = {'dataset': 'mario',
                'input_size': [100, 100, 3],
                'dataloader': MarioDataLoader,
                'num_sym_factor': 1,
                'sym_dim_list': [9],
                'sym_z_dim': 32,
                'subsym_z_dim': 32,
                'train_lr': 1e-4,
                'num_train_iteration': 5000,
                'abd_num_pos_case_per_bag': 5,
                'abd_num_neg_case_per_bag': 20,
                'abd_num_bag_per_batch': 8,
                'grounding_to_label_table': {
                                             (0, 0): 0,
                                             (1, 0): 1,
                                             (2, 0): 2,
                                             (2, 1): 3,
                                             (1, 1): 4,
                                             (0, 1): 5,
                                             (0, 2): 6,
                                             (1, 2): 7,
                                             (2, 2): 8},
                'grounding_to_label_table_str': {
                                                 '[0,0]': 0,
                                                 '[1,0]': 1,
                                                 '[2,0]': 2,
                                                 '[2,1]': 3,
                                                 '[1,1]': 4,
                                                 '[0,1]': 5,
                                                 '[0,2]': 6,
                                                 '[1,2]': 7,
                                                 '[2,2]': 8},
                'label_to_grounding_table': {-1: (-1, -1),
                                              0: (0, 0),
                                              1: (1, 0),
                                              2: (2, 0),
                                              3: (2, 1),
                                              4: (1, 1),
                                              5: (0, 1),
                                              6: (0, 2),
                                              7: (1, 2),
                                              8: (2, 2)},
                'integrity_table':
                    {8: ((3, 7), (2, 4, 6), (1, 5)),
                     7: ((4, 6, 8), (1, 3, 5), (0, 2)),
                     6: ((5, 7), (0, 4, 8), (1, 3))},
                'label_names': ['[0, 0]', '[1, 0]', '[2, 0]', '[2, 1]', '[1, 1]', '[0, 1]', '[0, 2]', '[1, 2]', '[2, 2]'],
                'abd_time_limit': 60,
                'recon_loss': torch.nn.MSELoss(reduction='sum'),
                'CE_loss': torch.nn.CrossEntropyLoss(reduction='none'),
                'BCE_loss': torch.nn.BCELoss(reduction='none')
}

