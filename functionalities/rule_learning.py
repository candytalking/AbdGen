"""
This file is a copyrighted under the BSD 3-clause licence, details of which can be found in the root directory.
Code for
Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings
https://arxiv.org/abs/2310.17451

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import print_function
import os, time, itertools
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.model import Model
from abd_functionals.abduction import AbdTrainer
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import pickle as pkl


# Suppress the warning
class TrainingProcessMonitor:
    def __init__(self, num_total_iterations, num_sym_factor, plot_gap=10):
        self.num_total_iterations, self.num_sym_factor = num_total_iterations, num_sym_factor
        self.plot_gap = plot_gap
        self.singleton_names = ['recon_loss']
        self.array_names = ['sym_loss', 'subsym_loss', 'vq_loss', 'sym_acc', 'subsym_acc', 'abd_acc', 'abd_rate']
        self.recorder = {}
        for name in self.singleton_names:
            self.recorder[name] = np.zeros(num_total_iterations)
        for name in self.array_names:
            self.recorder[name] = np.zeros([num_total_iterations, num_sym_factor])
        self.curr_iter = 0

    def record_info(self, info):
        for name in self.singleton_names:
            self.recorder[name][self.curr_iter] = info[name]
        for name in self.array_names:
            self.recorder[name][self.curr_iter] = info[name]

        if self.curr_iter % self.plot_gap == 0:
            mean_info = {}
            if self.curr_iter > 0:
                for name in self.singleton_names:
                    mean_info[name] = np.mean(self.recorder[name][self.curr_iter-self.plot_gap:self.curr_iter])
                for name in self.array_names:
                    mean_info[name] = np.mean(self.recorder[name][self.curr_iter-self.plot_gap:self.curr_iter], axis=0)
            else:
                for name in self.singleton_names:
                    mean_info[name] = self.recorder[name][0]
                for name in self.array_names:
                    mean_info[name] = self.recorder[name][0]

            print('============ curr iteration: %d =============\n'%self.curr_iter)
            for i in range(self.num_sym_factor):
                print('factor %d: sym loss: %.4f, subsym loss: %.4f, vq loss: %.4f, sym acc: %.4f, '
                      'subsym acc: %.4f, abd acc: %.4f, abd rate: %.4f'
                      %(i, mean_info['sym_loss'][i],  mean_info['subsym_loss'][i], mean_info['vq_loss'][i],
                        mean_info['sym_acc'][i], mean_info['subsym_acc'][i],
                        mean_info['abd_acc'][i], mean_info['abd_rate'][i]))
        self.curr_iter += 1


class NNSymGrounder:
    def __init__(self, config):
        self.num_fac = config['num_sym_factor']
        if config['dataset'] == 'mario':
            self.integrity_table = config['integrity_table']

    def update_bag_config(self, num_case_pos, num_case_neg, num_bag_per_batch):
        self.num_case_pos, self.num_case_neg = num_case_pos, num_case_neg
        self.num_bag = num_bag_per_batch
        self.num_inst_pos, self.num_inst_neg = self.num_case_pos * self.num_bag, self.num_case_neg * self.num_bag

    def get_groundings_mario(self, grd_pos, label_pos, vq_z, grounding_pred_softmax, grounding_label,
                             case_part_pos, num_pos, case_part_neg, batch_part_neg_per_instance):
        pos_nn_probs, pos_fac_names, neg_fac_names = [], [], []
        # get terminal information
        batch_part_pos, batch_part_neg, batch_term_pos, batch_term_label_pos \
            = self.data_into_inst(case_part_pos, case_part_neg, grd_pos, label_pos)
        pos_nn_probs, pos_fac_names \
            = self.get_pos_grds(grounding_pred_softmax, num_pos, case_part_pos, batch_term_label_pos)
        # get negative instance groundings
        for fac in range(self.num_fac):
            curr_vq_z = vq_z[fac]
            curr_pos_nn_label = np.split(grounding_label[fac][:num_pos].detach().cpu().numpy(), batch_part_pos, axis=0)
            curr_pos_fact_names = [pos_fac_names[i][fac] for i in range(self.num_bag)]
            tmp_pos_names = [list(itertools.chain.from_iterable(names)) for names in curr_pos_fact_names]
            neg_vq_dist = 2*(torch.sum(curr_vq_z ** 2, dim=1, keepdim=True) - torch.matmul(curr_vq_z, curr_vq_z.t()))[num_pos:, :num_pos]
            neg_vq_dist = np.split(neg_vq_dist.detach().to('cpu').numpy(), batch_part_neg, axis=0)
            neg_vq_dist = [np.split(neg_vq_dist[i], batch_part_pos, axis=1)[i] for i in range(self.num_bag)]
            neg_vq_dist = [np.split(neg_vq_dist[i], batch_part_neg_per_instance[i], axis=0) for i in range(self.num_bag)]
            min_dist_list = [[np.min(neg_vq_dist[i][j], axis=-1) for j in range(self.num_case_neg)] for i in range(self.num_bag)]
            argmin_list = [[[np.where(neg_vq_dist[i][j][k] == min_dist_list[i][j][k]) for k in range(min_dist_list[i][j].shape[0])]
                           for j in range(self.num_case_neg)] for i in range(self.num_bag)]
            argmin_label_list = [[[curr_pos_nn_label[i][argmin_list[i][j][k]] for k in range(min_dist_list[i][j].shape[0])]
                                 for j in range(self.num_case_neg)] for i in range(self.num_bag)]
            freq_argmin_label_list = [[[np.bincount(argmin_label_list[i][j][k]).argmax() for k in range(min_dist_list[i][j].shape[0])]
                                      for j in range(self.num_case_neg)] for i in range(self.num_bag)]
            curr_neg_fact_names = [[[tmp_pos_names[i][np.random.choice(np.where(curr_pos_nn_label[i] == freq_argmin_label_list[i][j][k])[0])]
                                     for k in range(len(freq_argmin_label_list[i][j]))] for j in range(self.num_case_neg)] for i in range(self.num_bag)]
            neg_fac_names.append(curr_neg_fact_names)
        neg_fac_names = [[neg_fac_names[j][i] for j in range(self.num_fac)] for i in range(self.num_bag)]
        return [(pos_nn_probs[i], batch_term_pos[i], pos_fac_names[i], neg_fac_names[i]) for i in range(len(pos_nn_probs))]

    def data_into_inst(self, case_part_pos, case_part_neg, grd_pos, label_pos):
        batch_part_pos = [case_part_pos[self.num_case_pos*(i+1)-1] for i in range(self.num_bag-1)]
        batch_part_neg = [case_part_neg[self.num_case_neg*(i+1)-1] for i in range(self.num_bag-1)]
        batch_term_pos = [[grd_pos[j][-1] for j in range(i * self.num_case_pos, (i + 1) * self.num_case_pos)]
                          for i in range(0, self.num_bag)]
        batch_term_label_pos = [[label_pos[j][-1] for j in range(i * self.num_case_pos, (i + 1) * self.num_case_pos)]
                                for i in range(0, self.num_bag)]
        return batch_part_pos, batch_part_neg, batch_term_pos, batch_term_label_pos

    def get_pos_grds(self, grounding_pred_softmax, num_pos, case_part_pos, term_pos):
        pos_nn_probs, pos_fac_names = [], []
        for fac in range(self.num_fac):
            curr_grounding_pred = grounding_pred_softmax[fac].detach().to('cpu').numpy()
            batched_curr_pred_pos = [np.split(curr_grounding_pred[:num_pos], case_part_pos, axis=0)[j:j+self.num_case_pos]
                                     for j in range(0, self.num_inst_pos, self.num_case_pos)]
            curr_pos_fact_names = [[['P_'+str(i)+'_'+str(j) for j in range(pred[i].shape[0])] for i in range(len(pred))]
                                   for pred in batched_curr_pred_pos]
            nn_prob_with_integrity = self.nn_prob_with_integrity(batched_curr_pred_pos, term_pos)
            pos_nn_probs.append(nn_prob_with_integrity)
            pos_fac_names.append(curr_pos_fact_names)
        pos_nn_probs = [[pos_nn_probs[j][i] for j in range(len(pos_nn_probs))] for i in range(len(pos_nn_probs[0]))]
        pos_fac_names = [[pos_fac_names[j][i] for j in range(len(pos_fac_names))] for i in range(len(pos_fac_names[0]))]
        return pos_nn_probs, pos_fac_names

    def nn_prob_with_integrity(self, nn_prob, term_pos):
        # integrity rule:
        # the start point should not within n-2 steps from the terminal
        label_ranking = 1
        nn_prob_with_integrity = []
        for i in range(self.num_bag):
            curr_nn_prob_with_integrity = []
            for j in range(self.num_case_pos):
                curr_nn_prob = nn_prob[i][j]
                curr_term_pos = term_pos[i][j]
                tmp_prob = np.copy(curr_nn_prob)
                tmp_prob[-1, :] = 0
                tmp_prob[-1, curr_term_pos] = 1
                tmp_prob[0, curr_term_pos] = 0
                for k in range(curr_nn_prob.shape[0]-3):
                    for h in self.integrity_table[curr_term_pos][k]:
                        tmp_prob[0, h] = 0
                if np.sum(tmp_prob[0]) == 0:
                    raise NameError('all probs are zero!')
                tmp_prob[0] = tmp_prob[0]/np.sum(tmp_prob[0])
                curr_nn_prob_with_integrity.append(tmp_prob)
            nn_prob_with_integrity.append(curr_nn_prob_with_integrity)
            haha = 1
        haha = 1
#        return nn_prob
        return nn_prob_with_integrity


def get_valid_train_labels(label_pos, abd_labels, num_bag_per_batch, num_pos_case_per_bag, batch_part_pos_per_bag):
    true_labels = [label_pos[i*num_pos_case_per_bag:(i+1)*num_pos_case_per_bag] for i in range(num_bag_per_batch)]
    valid_label_idx = [[np.ones(true_labels[i][j].shape[0]) for j in range(num_pos_case_per_bag)]
                       for i in range(num_bag_per_batch)]
    for i in range(num_bag_per_batch):
        if abd_labels[i] is not None:
            abd_labels[i] = np.split(np.asarray(abd_labels[i]), batch_part_pos_per_bag[i])
            for j in range(num_pos_case_per_bag):
                if abd_labels[i][j].shape[0] != np.unique(abd_labels[i][j]).shape[0]:
                    for k in range(num_pos_case_per_bag):
                        valid_label_idx[i][k][:] = 0
                    break
        else:
            abd_labels[i] = [np.zeros(true_labels[i][j].shape, dtype=np.int) # TODO: only for placeholder. Can be improved.
                             for j in range(len(true_labels[i]))]# true_labels[i]
            for j in range(num_pos_case_per_bag):
                valid_label_idx[i][j][:] = 0
    true_labels = np.concatenate([np.concatenate(true_labels[i]) for i in range(num_bag_per_batch)])
    abd_labels = np.concatenate([np.concatenate(abd_labels[i]) for i in range(num_bag_per_batch)])
    valid_label_idx = np.concatenate([np.concatenate(valid_label_idx[i]) for i in range(num_bag_per_batch)])
    return true_labels, [abd_labels], valid_label_idx


def train(config, path_manager, model_object):

    # --------------------- set data loader -------------------------
    dataloader = config['dataloader'](config, path_manager)
    dataset = config['dataset']
    # --------------------- set NN training -------------------------
    recon_criterion, CE_criterion, BCE_criterion = config['recon_loss'], config['CE_loss'], config['BCE_loss']
    device = torch.device("cuda:%s"%config['GPU'] if torch.cuda.is_available() else "cpu")
    config['device'] = device
    model_path = path_manager.get_spec_path('model')
    if model_object is not None:
        file_name = os.path.join(model_path, model_object['model_name'])
        model = torch.load(file_name)
    else:
        model = Model(config)
    model.to(device)
    lr = config['train_lr']
    betas = (.5, .9)
    a = list(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas) # amsgrad=False)
    scheduler = MultiStepLR(optimizer, milestones=[10000, 50000, 100000], gamma=0.5)
    train_monitor = TrainingProcessMonitor(config['num_train_iteration'], config['num_sym_factor'])
    # --------------------- set abductive process ------------------------
    abd_trainer = AbdTrainer(config=config, pl_tmp_path=path_manager.get_spec_path('pl'))
    nn_sym_grounder = NNSymGrounder(config)
    # ----------------------- training loop ------------------------------
    num_train_iteration = config['num_train_iteration']
    start_train_iteration = config['start_train_iteration']
    num_sym_factor = config['num_sym_factor']
    for train_itr in tqdm(range(start_train_iteration, num_train_iteration)):
        num_bag_per_batch = dataloader.num_bag_per_batch = config['abd_num_bag_per_batch']
        num_pos_case_per_bag = dataloader.num_pos_case_per_bag = config['abd_num_pos_case_per_bag']
        num_neg_case_per_bag = dataloader.num_neg_case_per_bag = config['abd_num_neg_case_per_bag']
        nn_sym_grounder.update_bag_config(num_pos_case_per_bag, num_neg_case_per_bag, num_bag_per_batch)
        num_case_per_batch_pos = dataloader.num_instance_per_batch_pos = num_bag_per_batch * num_pos_case_per_bag
        num_case_per_batch_neg = dataloader.num_instance_per_batch_neg = num_bag_per_batch * num_neg_case_per_bag
        if config['dataset'] == 'mario':
            img_pos, grd_pos, label_pos, batch_part_pos, batch_part_pos_per_bag, \
            img_neg, grd_neg, label_neg, batch_part_neg, batch_part_neg_per_bag = dataloader.get_train_batch()
            num_pos_instance, num_neg_instance = sum([img.shape[0] for img in img_pos]), sum([img.shape[0] for img in img_neg])
            imgs = torch.FloatTensor(np.concatenate(img_pos+img_neg, axis=0)).to(device).permute(0, 3, 1, 2)
            grd_pos_per_bag = np.split(grd_pos, batch_part_pos)
            grd_pos_per_bag = [grd_pos_per_bag[i*num_pos_case_per_bag:(i+1)*num_pos_case_per_bag]
                                for i in range(num_bag_per_batch)]
            label_pos_per_bag = np.split(label_pos, batch_part_pos)
            label_pos_per_bag = [label_pos_per_bag[i*num_pos_case_per_bag:(i+1)*num_pos_case_per_bag]
                                  for i in range(num_bag_per_batch)]

        # -------------------------------- model forward pass -----------------------------
        adv_alpha = 10
        sym_z, subsym_z, vq_z, vq_losses, grounding_pred, grounding_pred_softmax, grounding_label, adversarial_pred, \
        adversarial_pred_softmax, adversarial_label, adversarial_pred_detached, recon_x, all_recon_x \
            = model.forward(imgs, adv_alpha=adv_alpha)
        # ------------------------------ get groundings for abd ---------------------------
        grd_labels = torch.argmax(grounding_pred_softmax[0], dim=-1)[:num_pos_instance]
        adv_labels = torch.argmax(adversarial_pred_softmax[0], dim=-1)[:num_pos_instance]
        groundings_for_abd = nn_sym_grounder.get_groundings_mario(grd_pos, label_pos, vq_z,
                                                                  grounding_pred_softmax,
                                                                  grounding_label,
                                                                  batch_part_pos, num_pos_instance,
                                                                  batch_part_neg, batch_part_neg_per_bag)
        progs, abd_grds, abd_labels = abd_trainer.train_abduce(groundings_for_abd)
        true_labels, abd_labels, valid_label_idx \
            = get_valid_train_labels(label_pos, abd_labels, num_bag_per_batch, num_pos_case_per_bag, batch_part_pos_per_bag)
        confusion_matrix = np.zeros([9, 10])
        for k in range(9):
            haha_abd_labels = np.copy(abd_labels[0])
            haha_abd_labels[valid_label_idx == 0] = 9
            a = haha_abd_labels[true_labels == k]
            b = np.where(true_labels == k, haha_abd_labels, -np.ones(haha_abd_labels.shape))
            if k == 8 and sum(b == 0) != 0:
                haha = 1
            for k2 in range(10):
                confusion_matrix[k, k2] = sum(a == k2)
        print('\n confusion matrix:\n')
        print(confusion_matrix)

        abd_vq_losses = [torch.mean(torch.stack([vq_losses[j][abd_labels[j][i]][i] for i in range(num_pos_instance)], dim=0))
                         for j in range(num_sym_factor)]
        haha = 1
        num_effective_inst = np.sum(valid_label_idx)
        if np.prod(num_effective_inst) != 0:
            learn_flag = True
        else:
            learn_flag = False
        # ------------------------------ loss calculation -----------------------------------
        if learn_flag:
            # ---------------------------------------sym and subsym losses -----------------------------------
            grounding_pred[0] = [grounding_pred[0][i][:num_pos_instance] for i in range(config['num_code_heads'])]
            adversarial_pred[0] = adversarial_pred[0][:num_pos_instance]
            adversarial_pred_detached[0] = adversarial_pred_detached[0][:num_pos_instance]
            valid_label_idx = torch.tensor(valid_label_idx).to(torch.float32).to(device)
            abd_labels_one_hot = [F.one_hot(torch.tensor(abd_labels[i]), config['sym_dim_list'][i]).to(torch.float32).to(device)
                                  for i in range(num_sym_factor)]
            sym_losses = [torch.sum(torch.stack([torch.sum(CE_criterion(grounding_pred[i][j], abd_labels_one_hot[i]) * valid_label_idx)
                          for j in range(config['num_code_heads'])])) / num_effective_inst for i in range(num_sym_factor)]
            sym_loss = torch.sum(torch.stack(sym_losses))
            subsym_losses = [torch.sum(CE_criterion(adversarial_pred[i], abd_labels_one_hot[i]) * valid_label_idx) / num_effective_inst
                             for i in range(num_sym_factor)]
            subsym_losses_no_adv = [torch.sum(CE_criterion(adversarial_pred_detached[i], abd_labels_one_hot[i]) * valid_label_idx) / num_effective_inst
                                    for i in range(len(adversarial_pred_detached))]
            sym_losses_record = [sym_losses[i].detach().cpu().numpy() for i in range(len(sym_losses))]
            subsym_losses_record = [subsym_losses[i].detach().cpu().numpy() for i in range(len(subsym_losses))]
            clip_thresh = [1., 1.]   # 1.5
            train_subsym_losses = []
            for i in range(len(subsym_losses)):
                if subsym_losses[i].detach().cpu().numpy() < clip_thresh[i]:
                    train_subsym_losses.append(subsym_losses[i])
                else:
                    train_subsym_losses.append(subsym_losses_no_adv[i])
            subsym_loss = torch.sum(torch.stack(train_subsym_losses))
        else:
            # no valid abduced labels
            sym_losses_record, subsym_losses_record \
                = [0 for i in range(num_sym_factor)], [0 for i in range(num_sym_factor)]
            sym_loss, subsym_loss = 0, 0
        # --------------------------------------- recon and generator loss ---------------------------------------
        recon_loss = recon_criterion(recon_x, imgs) / (imgs.size(0) * 100)
        # -------------------------------------- overall generator update ---------------------------------------
        if learn_flag:
            if train_itr % 1 == 0:
                loss = 1 * recon_loss + 1 * sym_loss + 1 * torch.sum(torch.stack(abd_vq_losses)) + 1 * subsym_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        # ------------------------------------- record training process -----------------------------------------
            idces = valid_label_idx.detach().to('cpu').numpy()
            if config['dataset'] == 'mario':
                abd_rate = np.sum(idces)/idces.shape[0]
                sym_acc = np.sum((grd_labels.detach().to('cpu').numpy() == true_labels))/num_pos_instance
                subsym_acc = np.sum((adv_labels.detach().to('cpu').numpy() == true_labels))/num_pos_instance
                abd_acc = np.sum((abd_labels == true_labels)*idces)/np.sum(idces)
        else:
            abd_rate = 0
            sym_acc, subsym_acc, abd_acc = 0, 0, 0
        haha = 1

        # show and save results
        info = {'recon_loss': recon_loss.detach().to('cpu').numpy(),
                'sym_loss': sym_losses_record,
                'subsym_loss': subsym_losses_record,
                'vq_loss': [abd_vq_losses[i].detach().to('cpu').numpy() for i in range(len(abd_vq_losses))],
                'sym_acc': sym_acc,
                'subsym_acc': subsym_acc,
                'abd_acc': abd_acc,
                'abd_rate': abd_rate
                }
        train_monitor.record_info(info)
        if train_itr % 1 == 0:
            print("\nExample of Learned Programs:\n")
            print(progs[0])
        if train_itr == 0 or (train_itr + 1) % 100 == 0 or train_itr == config['num_train_iteration']-1:
            model_save_name = 'model.%d'%train_itr
            torch.save(model, os.path.join(model_path, model_save_name))
            result_save_path = path_manager.get_spec_path('result')
            with open(os.path.join(result_save_path, 'record_' + str(train_itr)), 'wb') as f:
                pkl.dump(train_monitor.recorder, f)



