"""
This file is a copyrighted under the BSD 3-clause licence, details of which can be found in the root directory.
Code for
Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings
https://arxiv.org/abs/2310.17451

"""

import os
import argparse
import exp_config.rule_learning_config as rule_learning_config
import functionalities.rule_learning as rule_learning
from utils.utils import set_seed, PathManager


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='run AbdGen')
    # =================== basic setup =======================
    parser.add_argument('--task', type=str, help='learning task', default='rule_learning')
    parser.add_argument('--dataset', type=str,
                        help='dataset to work on', default='mario')
    parser.add_argument('--exp_name', type=str,
                        help='name of experiments', default='exp_0')
    parser.add_argument('--GPU', type=str, help='# of GPU to use', default='0')
    parser.add_argument('--num_cpu_core', type=int, help='CPU core to use', default='16')
    parser.add_argument('--seed', type=int, help='random seed to use in the experiments', default=42)
    # =================== training setup ========================
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--model_name', type=str, help='name of the model to load', default='model.999')
    parser.add_argument('--start_train_iteration', type=int, help='start train iteration', default=0)
    # ==================== parameter setup ==========================
    parser.add_argument('--num_code_heads', type=int, help='number of heads in codebooks', default=1)
    # =================== path setup ========================
    parser.add_argument('--swipl', type=str,
                        help='location of swipl', default='/usr/bin/swipl')
    parser.add_argument('--bk_file', type=str,
                        help='location of background knowledge file, need to be absolute path',
                        default='/home/worker/code/AbdGen/prolog/mario_rule_learning_bk.pl')
    parser.add_argument('--exp_root_path', type=str,
                        help='exp root path', default='/home/worker/exp/AbdGen/')
    parser.add_argument('--dataset_path', type=str,
                        help='dataset path', default='dataset/')
    parser.add_argument('--model_path', type=str,
                        help='model path', default='model/')
    parser.add_argument('--tmp_path', type=str,
                        help='temp path', default='tmp/')
    parser.add_argument('--rule_path', type=str,
                        help='rule path', default='rule/')
    parser.add_argument('--result_path', type=str,
                        help='result path', default='result/')
    parser.add_argument('--pl_tmp_path', type=str,
                        help='prolog tmp file path', default='pl_tmp/')

    config_mapper = {'rule_learning': {'mario': rule_learning_config.mario_config},
                     }
    args = parser.parse_args()
    if args.seed > 0:
        set_seed(args.seed)
    config_object = config_mapper[args.task][args.dataset]
    config_object['num_cpu_core'] = args.num_cpu_core
    config_object['start_train_iteration'] = args.start_train_iteration
    config_object['GPU'] = args.GPU
    config_object['num_code_heads'] = args.num_code_heads
    config_object['swipl'] = args.swipl
    config_object['bk_file'] = args.bk_file
    path_object = {
                   'exp_root_path': args.exp_root_path,
                   'dataset_path': args.dataset_path,
                   'model_path': args.model_path,
                   'tmp_path': args.tmp_path,
                   'rule_path': args.rule_path,
                   'result_path': args.result_path,
                   'pl_tmp_path': args.pl_tmp_path
                   }
    path_manager = PathManager(path_object, args.dataset, args.exp_name)
    if args.load_model:
        model_object = {'model_name': args.model_name}
    else:
        model_object = None
    if args.task == 'rule_learning':
        rule_learning.train(config_object, path_manager, model_object)
    else:
        raise NameError('unrecognized task.')
