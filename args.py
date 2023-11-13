# -*- coding: utf-8 -*-
'''
    -------args.py-----------
    -----parameter configuration----
'''

import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())

from data.helper import seed_torch

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument('--dataset', type=str, default='MSTAR_FSL', help='Dataset name')
# 以miniimagenet数据集类型读取数据
parser.add_argument('--data_path', type=str,
                    default="../Datasets/MSTAR_FSL/",
                    help='The directory containing the mstar data.')
parser.add_argument('--train_mode', type=str, default='train17D',
                    help='meta train set.')
parser.add_argument('--test_mode1', type=str, default='test15D',
                    help='meta test set1.')          
# 以omniglot数据集类型读取数据                  
parser.add_argument('--train_data_dir', type=str,
                    default="../../Datasets/few-shot-learning/Omniglot/images_background/",
                    help='The directory containing the train image data.')
parser.add_argument('--val_data_dir', type=str,
                    default="../../Datasets/few-shot-learning/Omniglot/images_evaluation/",
                    help='The directory containing the validation image data.')
parser.add_argument('--summary_path', type=str,
                    default="./summary",
                    help='The directory of the summary writer.')

parser.add_argument('--n_way', type=int, default=3,
                    help='The number of class of every task.')
parser.add_argument('--k_shot', type=int, default=1,
                    help='The number of support set image for every task.')
parser.add_argument('--q_query_train', type=int, default=15,
                    help='The number of query set image for train task.')
parser.add_argument('--q_query_test', type=int, default=15,
                    help='The number of query set image for test task.')                    
parser.add_argument('--in_channels', type=int, default=3,
                    help='The number of input channles.')
parser.add_argument('--resize', type=int, default=64,
                    help='resize the image size.')                      
parser.add_argument('--epochs', type=int, default=2,
                    help='The training epochs.')
parser.add_argument('--lamda', type=float, default=9e-4,
                    help='The learning rate of of the support set.')                    
parser.add_argument('--train_episode_num', type=int, default=4000,
                    help='Number of train_episodes.')
parser.add_argument('--test_episode_num', type=int, default=300,
                    help='Number of test_episodes.')                    
parser.add_argument('--train_batch_size', type=int, default=10,
                    help='Number of task per train batch.')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='Number of task per test batch.')
parser.add_argument('--train_inner_step', type=int, default=1,
                    help='Number of train inner steps.')
parser.add_argument('--test_inner_step', type=int, default=5,
                    help='Number of test inner steps.')
parser.add_argument('--merge_inner_step', type=int, default=1,
                    help='Number of merge inner steps that considers multiple inner steps initial parameters.')                    
parser.add_argument('--adaptiveSGD', type=bool, default=False,
                    help='whether to use adaptiveSGD?')                    
parser.add_argument('--algorithm', type=str, default='Mada-SGD', 
                    help='FSL algorithms: MAML, Meta-SGD and Mada-SGD')
parser.add_argument('--backbone', type=str, default='ConvNet4', 
                    help='feature embedding model: ConvNet4, ResNet8 and GCN')                    
parser.add_argument('--num_workers', type=int, default=0, 
                    help='The number of torch dataloader thread.')
                   

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
seed_torch(1206)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
