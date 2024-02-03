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
from torch.autograd import Function

import torch
import numpy as np
import torch.nn as nn

# from models.building_blocks import *


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



class AdversarialClassifier(nn.Module):
    def __init__(self, device, input_dim, output_dim, use_batch_norm=True):
        super(AdversarialClassifier, self).__init__()
        hidden_dim = 512
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.classifier = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.act = nn.ReLU(inplace=True)
        if use_batch_norm:
            self.use_batch_norm = True
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        else:
            self.use_batch_norm = False

    def forward(self, embeddings, alpha):
        reversed_embeddings = ReverseLayerF.apply(embeddings, alpha)
        z, z_detached = self.fc1(reversed_embeddings), self.fc1(embeddings.detach())
        if self.use_batch_norm:
            z, z_detached = self.bn1(z), self.bn1(z_detached)
        z, z_detached = self.fc2(self.act(z)), self.fc2(self.act(z_detached))
        if self.use_batch_norm:
            z, z_detached = self.bn2(z), self.bn2(z_detached)
        z, z_detached = self.act(z), self.act(z_detached)
        preds, preds_detached = self.classifier(z), self.classifier(z_detached) # .to(torch.float32)
        return preds, preds_detached


class VQModule(nn.Module):

    def __init__(self, device, input_dim, dict_dim, num_factor, per_factor_dict_size, commitment_cost=0.25):
        super(VQModule, self).__init__()
        self.device = device
        self.embedding_dim = dict_dim
        self.input_dim = input_dim
        self.num_factor = num_factor
        self.per_factor_dict_size = per_factor_dict_size
        self.num_embeddings = num_factor
        # Feed in codebook vectors
        self.embedding = nn.ModuleList([nn.Embedding(self.num_factor, self.embedding_dim).to(self.device) for i in range(self.per_factor_dict_size)])
        self.linear1 = nn.ModuleList([nn.Linear(self.input_dim, 512).to(self.device) for i in range(self.per_factor_dict_size)])
        self.linear2 = nn.ModuleList([nn.Linear(512, self.embedding_dim).to(self.device) for i in range(self.per_factor_dict_size)])
        self.act = nn.ReLU(inplace=True)
        for i in range(self.per_factor_dict_size):
            self.embedding[i].weight.data.uniform_(-1/ self.num_embeddings, 1 / self.num_embeddings)
        self.commitment_cost = 0.25 #

    def forward(self, inputs, label_gen=None, testing=False):
        # Flatten input
        input_z = inputs.contiguous() # .view(-1, self.input_dim)
        # Calculate distances
        multi_input_z = [self.linear2[i](self.act(self.linear1[i](input_z))) for i in range(self.per_factor_dict_size)]
        distances = [(torch.sum(multi_input_z[i] ** 2, dim=1, keepdim=True) + torch.sum(self.embedding[i].weight ** 2, dim=1)
                     - 2 * torch.matmul(multi_input_z[i], self.embedding[i].weight.t())) for i in range(self.per_factor_dict_size)]
        ave_distances = torch.mean(torch.concat([distances[i].unsqueeze(1)
                                                 for i in range(self.per_factor_dict_size)], dim=1), dim=1)
        # find closest encodings
        min_encoding_indices = torch.argmin(ave_distances, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_factor).to(self.device)
        min_encodings = min_encodings.scatter_(1, min_encoding_indices, 1)
        vq_z = [torch.matmul(min_encodings, self.embedding[i].weight).view(multi_input_z[i].shape) for i in range(self.per_factor_dict_size)]
        vq_z = torch.mean(torch.concat([vq_z[i].unsqueeze(1) for i in range(self.per_factor_dict_size)], dim=1), dim=1)
        vq_z = input_z + (vq_z - input_z).detach()
        haha = 1
        #  -------------------- return all vq_results during training -------------------------
        if not testing:
            all_vq_z = [torch.mean(torch.concat([self.embedding[i].weight[j].unsqueeze(1)
                                                 for i in range(self.per_factor_dict_size)], dim=1), dim=1)
                        for j in range(self.num_factor)]
            all_vq_loss = [torch.sum((all_vq_z[i]- input_z.detach()) ** 2, dim=1) \
                           + self.commitment_cost * torch.sum((all_vq_z[i].detach() - input_z) ** 2, dim=1)
                           for i in range(self.num_factor)]
            all_vq_z = [input_z + (all_vq_z[i] - input_z).detach() for i in range(self.num_factor)]
        else:
            all_vq_z, all_vq_loss = None, None
#        # ----------- changed generation -------------
        if label_gen is not None:
            changed_vq_z = torch.mean(torch.concat([self.embedding[i].weight[label_gen].unsqueeze(1)
                                                    for i in range(self.per_factor_dict_size)], dim=1), dim=1)
        else:
            changed_vq_z = None
        return vq_z, distances, all_vq_z, all_vq_loss, changed_vq_z


class CNNEncoder(nn.Module):

    def __init__(self, input_size, output_dim, use_batch_norm=False):
        super(CNNEncoder, self).__init__()
        self.use_batch_norm = use_batch_norm
        kernel_size = (4, 4, 4, 4)
        strides = (2, 2, 2, 2)
        n_channels = (16, 32, 64, 128)
        self.output_height_list = np.zeros(4, dtype=np.int)
        self.padding_list = np.zeros(4, dtype=np.int)
        self.input_size, self.output_dim = input_size, output_dim
        self.output_height_list[0] = int(np.floor(self.input_size[0]/strides[0]))
        self.padding_list[0] = np.ceil(((self.output_height_list[0]-1)*strides[0]-self.input_size[0]+kernel_size[0])/2)
        self.conv0 = nn.Conv2d(3, n_channels[0], kernel_size[0], strides[0], self.padding_list[0])  # 50 x 50
        self.bn0 = nn.BatchNorm2d(n_channels[0])

        self.output_height_list[1] = int(np.floor(self.output_height_list[0]/strides[1]))
        self.padding_list[1] = np.ceil(((self.output_height_list[1]-1)*strides[1]-self.output_height_list[0]+kernel_size[1])/2)
        self.conv1 = nn.Conv2d(n_channels[0], n_channels[1], kernel_size[1], strides[1], self.padding_list[1])  # 25 x 25
        self.bn1 = nn.BatchNorm2d(n_channels[1])

        self.output_height_list[2] = int(np.floor(self.output_height_list[1]/strides[2]))
        self.padding_list[2] = np.ceil(((self.output_height_list[2]-1)*strides[2]-self.output_height_list[1]+kernel_size[2])/2)
        self.conv2 = nn.Conv2d(n_channels[1], n_channels[2], kernel_size[2], strides[2], self.padding_list[2])  # 12 x 12
        self.bn2 = nn.BatchNorm2d(n_channels[2])

        self.output_height_list[3] = int(np.floor(self.output_height_list[2]/strides[3]))
        self.padding_list[3] = np.ceil(((self.output_height_list[3]-1)*strides[3]-self.output_height_list[2]+kernel_size[3])/2)
        self.conv3 = nn.Conv2d(n_channels[2], n_channels[3], kernel_size[3], strides[3], self.padding_list[3])  # 6 x 6
        self.bn3 = nn.BatchNorm2d(n_channels[3])

        self.act = nn.ReLU(inplace=True)
        self.after_conv_shape = (128, self.output_height_list[3], self.output_height_list[3])
        self.after_conv_dim = self.output_height_list[3]**2*128
        self.linear_transform = nn.Linear(self.after_conv_dim, self.output_dim)


    def forward(self, x):
        h = self.act(self.conv0(x))
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = h.reshape(-1, self.after_conv_dim)
        h = self.linear_transform(h)
        return h

    def get_paired_decoder_param_list(self):
        target_output_list = [self.output_height_list[i] for i in range(2, -1, -1)] + [self.input_size[0]]
        return self.after_conv_shape, target_output_list


class CNNDecoder(nn.Module):
    def __init__(self, input_dim, init_shape, output_list):
        super(CNNDecoder, self).__init__()

        self.init_shape = init_shape
        init_dim = int(np.prod(init_shape))
        self.linear = nn.Linear(input_dim, init_dim)
        padding, output_padding=1, 0
        self.deconv0 = nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), padding=padding, output_padding=output_padding)  # 12 x 12
        self.bn0 = nn.BatchNorm2d(64)
        padding, output_padding=1, 1
        self.deconv1 = nn.ConvTranspose2d(64, 32, (4, 4), (2, 2), padding=padding, output_padding=output_padding)  # 24 x 24
        self.bn1 = nn.BatchNorm2d(32)
        padding, output_padding=1, 0
        self.deconv2 = nn.ConvTranspose2d(32, 16, (4, 4), (2, 2), padding=padding, output_padding=output_padding)  # 50 x 50
        self.bn2 = nn.BatchNorm2d(16)
        padding, output_padding=1, 0
        self.deconv3 = nn.ConvTranspose2d(16, 3, (4, 4), (2, 2), padding=padding, output_padding=output_padding)  # 100 x 100
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.linear(x).view(-1, self.init_shape[0], self.init_shape[1], self.init_shape[2])
        h = self.act(self.deconv0(h))
        h = self.act(self.deconv1(h))
        h = self.act(self.deconv2(h))
        h = self.deconv3(h)
        watch = h.detach().cpu().numpy()
        return h


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.device = config['device']
        self.input_size = config['input_size']
        self.sym_z_dim, self.subsym_z_dim, self.num_code_heads = config['sym_z_dim'], config['subsym_z_dim'], config['num_code_heads']
        self.num_sym_factor = config['num_sym_factor']
        self.sym_dim_list = config['sym_dim_list']
        self.sym_output_dim = sum(self.sym_dim_list)
        self.num_code_heads = config['num_code_heads']
        self.decoder_input_dim = self.sym_z_dim*self.num_sym_factor + self.subsym_z_dim
        self.sym_encoder = CNNEncoder(self.input_size, self.sym_z_dim)
        self.subsym_encoder = CNNEncoder(self.input_size, self.subsym_z_dim)
        decoder_init_shape, padding_list = self.sym_encoder.get_paired_decoder_param_list()
        self.decoder = CNNDecoder(self.decoder_input_dim, decoder_init_shape, padding_list)
        self.adversarial_classifier_list = nn.ModuleList([AdversarialClassifier(self.device, self.subsym_z_dim, self.sym_dim_list[i])
                                                          for i in range(self.num_sym_factor)])
        self.vq_module = nn.ModuleList([VQModule(self.device, self.sym_z_dim, self.sym_z_dim, self.sym_dim_list[i], self.num_code_heads)
                                        for i in range(self.num_sym_factor)])


    def forward(self, x, adv_alpha=1, only_recon=False, testing=False):

        sym_z = self.sym_encoder(x)
        subsym_z = self.subsym_encoder(x)
        vq_result = [self.vq_module[i](sym_z, testing=testing) for i in range(self.num_sym_factor)]
        vq_z = [vq_result[i][0] for i in range(self.num_sym_factor)]
        all_vq_z = [vq_result[i][2] for i in range(self.num_sym_factor)]
        decoder_input_z = torch.cat([torch.cat(vq_z, dim=1), subsym_z], dim=1)
        recon_x = self.decoder(decoder_input_z)
        if not testing:
            if self.num_sym_factor == 1:
                all_decoder_input_z = torch.stack([torch.cat([all_vq_z[0][j], subsym_z], dim=1) for j in range(self.sym_dim_list[0])])
                all_recon_x = self.decoder(all_decoder_input_z)
                all_recon_x = all_recon_x.view([self.sym_dim_list[0], all_decoder_input_z.size(1),
                                                all_recon_x.size(-3), all_recon_x.size(-2), all_recon_x.size(-1)])
            elif self.num_sym_factor == 2:
                all_decoder_input_z = torch.stack([torch.stack([torch.cat([torch.cat([all_vq_z[0][i], all_vq_z[1][j]], dim=1), subsym_z], dim=1)
                                                   for j in range(self.sym_dim_list[1])]) for i in range(self.sym_dim_list[0])])
                all_recon_x = self.decoder(all_decoder_input_z)
                all_recon_x = all_recon_x.view([self.sym_dim_list[0], self.sym_dim_list[1], all_decoder_input_z.size(2),
                                                all_recon_x.size(-3), all_recon_x.size(-2), all_recon_x.size(-1)])
            else:
                raise NameError('unsupported num sym factor!')  # TODO: implement general situation
        else:
            all_recon_x = None
        haha = 1
        if only_recon:
            return recon_x
        else:
            vq_losses = [vq_result[i][3] for i in range(self.num_sym_factor)]
            distances = [vq_result[i][1] for i in range(self.num_sym_factor)]
            t = 1
            grounding_pred = [[-distances[i][j] / t for j in range(self.num_code_heads)] for i in range(self.num_sym_factor)]
            mean_grounding_pred = [torch.mean(torch.concat([grounding_pred[i][j].unsqueeze(1)
                                                            for j in range(self.num_code_heads)], dim=1), dim=1)
                                   for i in range(self.num_sym_factor)]
            mean_grounding_pred_softmax = [torch.softmax(mean_grounding_pred[i], dim=1) + 1e-3
                                           for i in range(self.num_sym_factor)]
            adversarial_pred_full = [self.adversarial_classifier_list[i](subsym_z, adv_alpha)
                                     for i in range(self.num_sym_factor)]
            adversarial_pred = [adversarial_pred_full[i][0] for i in range(self.num_sym_factor)]
            adversarial_pred_detached = [adversarial_pred_full[i][1] for i in range(self.num_sym_factor)]
            adversarial_pred_softmax = [torch.softmax(adversarial_pred[i], dim=1) for i in range(self.num_sym_factor)]
            grounding_label = [torch.argmax(mean_grounding_pred[i], dim=1) for i in range(self.num_sym_factor)]
            adversarial_label = [torch.argmax(adversarial_pred[i], dim=1) for i in range(self.num_sym_factor)]

            return sym_z, subsym_z, vq_z, vq_losses,\
                   grounding_pred, mean_grounding_pred_softmax, grounding_label, \
                   adversarial_pred, adversarial_pred_softmax, adversarial_label, adversarial_pred_detached, \
                   recon_x, all_recon_x

    def conditional_generation(self, x, c_idx):
        sym_z = self.sym_encoder(x)
        subsym_z = self.subsym_encoder(x)
        vq_result = [self.vq_module[i](sym_z, label_gen=c_idx[:, i], testing=True) for i in range(self.num_sym_factor)]
        changed_vq_z = [vq_result[i][4] for i in range(self.num_sym_factor)]
        decoder_input_z = torch.cat([torch.cat(changed_vq_z, dim=1), subsym_z], dim=1)
        watch = decoder_input_z.detach().cpu().numpy()
        recon_x = self.decoder(decoder_input_z)
        return recon_x


