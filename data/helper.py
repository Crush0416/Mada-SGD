# -*- coding: utf-8 -*-




from networks.conv4 import ConvNet4
from networks.resnet8 import ResNet8
from data.dataset import OmniglotDataset
from data.dataset.MiniImagenet import MiniImagenet

import os
import torch
from torch import nn
import numpy as np
import random


def get_model(args, device):
    """
    Get model.
    Args:
        args: ArgumentParser
        device: torch device

    Returns: model

    """
    if args.backbone == 'ConvNet4':
        model = ConvNet4(args.in_channels, args.n_way).cuda()
        model.to(device)
    if args.backbone == 'ResNet8':
        model = ResNet8(args.in_channels, args.n_way, out_channels=64).cuda()
        model.to(device)
    if args.backbone == 'GCN':
        model = GCN(args.in_channels, args.n_way).cuda()
        model.to(device)
    
    return model

def get_dataset_miniimagenet(args):
    """
    Get FSL dataset.   miniimagenet dataset type
    Args:
        args: ArgumentParser

    Returns: dataset

    """
    train_dataset = MiniImagenet(root=args.data_path, mode=args.train_mode,
                                n_way=args.n_way, k_shot=args.k_shot, k_query=args.q_query_train,
                                episode_num=args.train_episode_num, resize=args.resize)
    
    test_dataset = MiniImagenet(root=args.data_path, mode=args.test_mode1,
                                 n_way=args.n_way, k_shot=args.k_shot, k_query=args.q_query_test,
                                 episode_num=args.test_episode_num, resize=args.resize)
    
    # test_dataset2 = MiniImagenet(root=args.data_path, mode=args.test_mode2,
                             # n_way=args.n_way, k_shot=args.k_shot, k_query=args.q_query_test,
                             # episode_num=args.test_episode_num, resize=args.resize)
    
    return train_dataset, test_dataset
    
    
def get_dataset_omniglot(args):
    """
    Get FSL dataset.
    Args:
        args: ArgumentParser

    Returns: dataset

    """
    train_dataset = OmniglotDataset(args.train_data_dir, args.task_num,
                                    n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
    val_dataset = OmniglotDataset(args.val_data_dir, args.val_task_num,
                                  n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)

    return train_dataset, val_dataset


def seed_torch(seed):
    """
    Set all random seed
    Args:
        seed: random seed

    Returns: None

    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def remove_dir_and_create_dir(dir_name, is_remove=True):
    """
    Make new folder, if this folder exist, we will remove it and create a new folder.
    Args:
        dir_name: path of folder
        is_remove: if true, it will remove old folder and create new folder

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "create.")
    else:
        if is_remove:
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
            print(dir_name, "create.")
        else:
            print(dir_name, "is exist.")