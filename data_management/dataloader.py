"""
This file is a copyrighted under the BSD 3-clause licence, details of which can be found in the root directory.
Code for
Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings
https://arxiv.org/abs/2310.17451

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class MarioDataLoader:
    def __init__(self, config, path_manager):
        self.dataset = config['dataset']
        self.grounding_to_label_table = config['grounding_to_label_table']
        pos_data_path = path_manager.get_gen_file('dataset', self.dataset+'_pos.npz')
        self.pos_data_zip = np.load(pos_data_path, encoding='latin1', allow_pickle=True)
        self.imgs_pos_pri, self.grds_pos_pri = self.pos_data_zip['images'], self.pos_data_zip['pos']
        # avoid to use straight line paths
        tmp_idx1 = [i for i in range(len(self.imgs_pos_pri)) if self.grds_pos_pri[i] == [(1, 0), (1, 1), (1, 2)]]
        tmp_idx2 = [i for i in range(len(self.imgs_pos_pri)) if self.grds_pos_pri[i] == [(2, 0), (2, 1), (2, 2)]]
        self.imgs_pos = [self.imgs_pos_pri[i] for i in range(len(self.imgs_pos_pri)) if (i not in tmp_idx1 and i not in tmp_idx2)]
        self.num_pos_data = len(self.imgs_pos)
        self.grds_pos = [self.grds_pos_pri[i] for i in range(len(self.grds_pos_pri)) if (i not in tmp_idx1 and i not in tmp_idx2)]
        self.grds_pos = [np.concatenate([np.expand_dims(j, axis=0) for j in i]) for i in self.grds_pos]
        self.imgs_pos = [self.imgs_pos[i][:self.grds_pos[i].shape[0]] for i in range(self.num_pos_data)]
        self.labels_pos = [np.asarray([self.grounding_to_label_table[tuple(j)] for j in i]) for i in self.grds_pos]
        self.long_pos_case_idx = [np.asarray([idx for idx in range(len(self.labels_pos)) if self.labels_pos[idx].shape[0]==5]),
                                  np.asarray([idx for idx in range(len(self.labels_pos)) if self.labels_pos[idx].shape[0]==4])]
        self.terms_pos = np.asarray([label[-1] for label in self.labels_pos])
        neg_data_path = path_manager.get_gen_file('dataset', self.dataset+'_neg.npz')
        self.neg_data_zip = np.load(neg_data_path, encoding='latin1', allow_pickle=True)
        self.imgs_neg_pri, self.grds_neg_pri = list(self.neg_data_zip['images']), list(self.neg_data_zip['pos'])

        # ensure to have enough useful negative cases. Constructed before learning.
        self.aux_imgs = [[] for i in range(9)]
        for i in range(len(self.imgs_neg_pri)):
            for j in range(len(self.imgs_neg_pri[i])):
                if self.grds_neg_pri[i][j] == (0, 0):
                    self.aux_imgs[0].append(self.imgs_neg_pri[i][j])
                elif self.grds_neg_pri[i][j] == (1, 0):
                    self.aux_imgs[1].append(self.imgs_neg_pri[i][j])
                elif self.grds_neg_pri[i][j] == (2, 0):
                    self.aux_imgs[2].append(self.imgs_neg_pri[i][j])
                elif self.grds_neg_pri[i][j] == (2, 1):
                    self.aux_imgs[3].append(self.imgs_neg_pri[i][j])
                elif self.grds_neg_pri[i][j] == (1, 1):
                    self.aux_imgs[4].append(self.imgs_neg_pri[i][j])
                elif self.grds_neg_pri[i][j] == (0, 1):
                    self.aux_imgs[5].append(self.imgs_neg_pri[i][j])
                elif self.grds_neg_pri[i][j] == (0, 2):
                    self.aux_imgs[6].append(self.imgs_neg_pri[i][j])
                elif self.grds_neg_pri[i][j] == (1, 2):
                    self.aux_imgs[7].append(self.imgs_neg_pri[i][j])
                elif self.grds_neg_pri[i][j] == (2, 2):
                    self.aux_imgs[8].append(self.imgs_neg_pri[i][j])

        self.aux_imgs = [np.unique(np.asarray(self.aux_imgs[i]), axis=0) for i in range(9)]
        case1 = [(self.aux_imgs[5][i], self.aux_imgs[4][i], self.aux_imgs[7][i]) for i in range(168)]
        grds1 = [[(0, 1), (1, 1), (1, 2)] for i in range(168)]
        case2 = [(self.aux_imgs[0][i], self.aux_imgs[5][i], self.aux_imgs[4][i], self.aux_imgs[7][i]) for i in range(168)]
        grds2 = [[(0, 0), (0, 1), (1, 1), (1, 2)] for i in range(168)]
        case3 = [(self.aux_imgs[0][i], self.aux_imgs[5][i], self.aux_imgs[4][i], self.aux_imgs[7][i], self.aux_imgs[8][i]) for i in range(168)]
        grds3 = [[(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)] for i in range(168)]
        case4 = [(self.aux_imgs[0][i], self.aux_imgs[5][i], self.aux_imgs[4][i], self.aux_imgs[3][i], self.aux_imgs[8][i]) for i in range(168)]
        grds4 = [[(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)] for i in range(168)]
        case5 = [(self.aux_imgs[0][i], self.aux_imgs[1][i], self.aux_imgs[4][i], self.aux_imgs[7][i], self.aux_imgs[8][i]) for i in range(168)]
        grds5 = [[(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)] for i in range(168)]
        case6 = [(self.aux_imgs[0][i], self.aux_imgs[1][i], self.aux_imgs[4][i], self.aux_imgs[3][i], self.aux_imgs[8][i]) for i in range(168)]
        grds6 = [[(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)] for i in range(168)]
        self.imgs_neg_pri = self.imgs_neg_pri + case1 + case2 + case3 + case4 + case5 + case6
        self.grds_neg_pri = self.grds_neg_pri + grds1 + grds2 + grds3 + grds4 + grds5 + grds6
        tmp_idx1 = [i for i in range(len(self.imgs_neg_pri)) if self.grds_neg_pri[i] == [(1, 1), (2, 1), (2, 2)]]
        tmp_idx2 = [i for i in range(len(self.imgs_neg_pri)) if self.grds_neg_pri[i] == [(0, 1), (1, 1), (1, 2)]]
        self.imgs_neg = [self.imgs_neg_pri[i] for i in range(len(self.imgs_neg_pri)) if (i not in tmp_idx1 and i not in tmp_idx2)]
        self.grds_neg = [self.grds_neg_pri[i] for i in range(len(self.grds_neg_pri)) if (i not in tmp_idx1 and i not in tmp_idx2)]
        self.imgs_neg = [np.concatenate(np.expand_dims(imgs, axis=0), axis=0) for imgs in self.imgs_neg]
        self.num_neg_data = len(self.imgs_neg)
        self.grds_neg = [np.concatenate([np.expand_dims(j, axis=0) for j in i]) for i in self.grds_neg]
        self.labels_neg = [np.asarray([self.grounding_to_label_table[tuple(j)] for j in i]) for i in self.grds_neg]
        self.long_neg_case_idx = [np.asarray([idx for idx in range(len(self.labels_neg)) if self.labels_neg[idx].shape[0]==5])]

        self.train_img_mean = np.mean(np.concatenate([np.concatenate(self.imgs_pos, axis=0),
                                                      np.concatenate(self.imgs_neg, axis=0)], axis=0), axis=0)
        self.train_img_std = np.std(np.concatenate([np.concatenate(self.imgs_pos, axis=0),
                                                    np.concatenate(self.imgs_neg, axis=0)], axis=0), axis=0)


    def get_train_batch(self):
        self.num_pos = self.num_bag_per_batch*self.num_pos_case_per_bag
        self.num_neg = self.num_bag_per_batch*self.num_neg_case_per_bag
        idx_five = np.random.choice(self.long_pos_case_idx[0], self.num_bag_per_batch, replace=False).reshape([-1, 1])
        idx_four = np.random.choice(self.long_pos_case_idx[1], self.num_bag_per_batch, replace=False).reshape([-1, 1])
        idx_other = np.random.choice(self.num_pos_data, self.num_pos-self.num_bag_per_batch*2,
                                     replace=False).reshape([-1, self.num_pos_case_per_bag-2])
        idx_pos = np.concatenate([idx_five, idx_four, idx_other], axis=-1).reshape([-1])
        img_pos = [(self.imgs_pos[i].astype(np.float32)-self.train_img_mean)/self.train_img_std for i in idx_pos]
        grd_pos, label_pos = [self.grds_pos[i] for i in idx_pos], [self.labels_pos[i] for i in idx_pos]
        len_pos = [grd_pos[i].shape[0] for i in range(self.num_pos)]
        batch_part_pos = [sum(len_pos[:i+1]) for i in range(self.num_pos-1)]
        len_pos_per_bag = [len_pos[i*self.num_pos_case_per_bag:(i+1)*self.num_pos_case_per_bag]
                           for i in range(self.num_bag_per_batch)]
        batch_part_pos_per_bag = [[sum(len_pos_per_bag[i][:j+1]) for j in range(self.num_pos_case_per_bag-1)]
                                   for i in range(self.num_bag_per_batch)]
        idx_five = np.random.choice(self.long_neg_case_idx[0], self.num_bag_per_batch*10, replace=False).reshape([-1, 10])
        idx_other = np.random.choice(self.num_neg_data, self.num_neg-self.num_bag_per_batch*10, replace=False).reshape([-1, self.num_neg_case_per_bag-10])
        idx_neg = np.concatenate([idx_five, idx_other], axis=-1).reshape([-1])
        img_neg = [(self.imgs_neg[i].astype(np.float32)-self.train_img_mean)/self.train_img_std for i in idx_neg]
        grd_neg, label_neg = [self.grds_neg[i] for i in idx_neg], [self.labels_neg[i] for i in idx_neg]
        len_neg = [grd_neg[i].shape[0] for i in range(self.num_neg)]
        batch_part_neg = [sum(len_neg[:i+1]) for i in range(self.num_neg-1)]
        len_neg_per_bag = [len_neg[i*self.num_neg_case_per_bag:(i+1)*self.num_neg_case_per_bag]
                           for i in range(self.num_bag_per_batch)]
        batch_part_neg_per_instance = [[sum(len_neg_per_bag[i][:j+1]) for j in range(self.num_neg_case_per_bag-1)]
                                       for i in range(self.num_bag_per_batch)]
        return img_pos, grd_pos, label_pos, batch_part_pos, batch_part_pos_per_bag, \
               img_neg, grd_neg, label_neg, batch_part_neg, batch_part_neg_per_instance


