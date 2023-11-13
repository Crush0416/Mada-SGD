# -*- coding: utf-8 -*-




import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from args import args, device
from data.helper import get_model, get_dataset_omniglot, get_dataset_miniimagenet
from algorithms.meta_sgd import MetaSGD, meta_sgd_loop
from algorithms.maml import maml_loop
from algorithms.mada_sgd import adaptiveSGD, adaptiveSGD2, adaptiveSGD3, mada_sgd_loop
import copy
import scipy.io
from sklearn.metrics import confusion_matrix

from data.dataset.MiniImagenet import MiniImagenet

if __name__ == '__main__':
    
    # -----------load meta-model----------
    path = './results/Mada/layer-wise/step1/17D_30D/3way_5shot/' 
    name = 'MSTAR_FSL-ConvNet4-Mada-SGD-3way-5shot-epoch-1-batch-250-0.9470acc.pt'

    Mada_SGD = torch.load(path+name)
    model = Mada_SGD['base_learner']
    mada_sgd = Mada_SGD['algorithm']
    test_acc =  Mada_SGD['val_acc']
    predict1 =  Mada_SGD['val_predict']
    label1 =  Mada_SGD['val_label']
    
    print('-' * 100)
    print('algorithm: {},  train inner steps: {},  test_inner step:{},  backbone: {},  paras num:{} K'
          .format(args.algorithm, args.train_inner_step, args.test_inner_step, args.backbone, 
          sum(p.numel() for p in model.parameters() if p.requires_grad)/1000))
    
    #  按照Omniglot数据集类型导入数据
    '''
    train_dataset, val_dataset = get_dataset_omniglot(args)

    train_loader = DataLoader(train_dataset, batch_size=args.task_num, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.val_task_num, shuffle=False, num_workers=args.num_workers)
    '''
    #  按照MiniImageNet数据集类型导入数据
    args.test_mode1 = 'test15D'
    args.k_shot = 5
    args.q_query_test = 269       # 1-shot: 15D-273, 17D-297, 30D-286, 45D-302
                                  # 5-shot: 15D-269, 17D-293, 30D-282, 45D-298
    args.test_episode_num = 10
    train_dataset, test_dataset = get_dataset_miniimagenet(args)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)
    
    print('-' * 100)
    print('train image num: {}, train image class: {}, test image num: {}, test image class: {}'.format(len([img for imgs in train_dataset.data for img in imgs]), train_dataset.cls_num,
                                                                                                        len([img for imgs in test_dataset.data for img in imgs]), test_dataset.cls_num))

    print('-' * 100)  

    label = [item.cpu().detach().numpy() for item in label1]
    label = np.array(label).flatten()
    predict = np.array(predict1).flatten()
    matrix = confusion_matrix(label, predict)
    print('############ confusion matrix ########### \n', matrix / 15 * 287 / 300)
    print("=> Min_Acc: {:.4f}   Max_acc: {:.4f}   Mean_Acc: {:.4f}   Std: {:.6f}".
          format(np.min(test_acc), np.max(test_acc), np.mean(test_acc), np.std(test_acc)))
    
    #   保存 test_acc
    save_path = path + 'test_acc_distribution.mat'
    scipy.io.savemat(save_path, {'test_acc':test_acc})
   
    model.eval()
    val_acc = []
    val_loss = []
    val_predict = []
    val_label = []
    for support_images, support_labels, query_images, query_labels in test_loader:

        # Get variables
        support_images = support_images.float().to(device)
        support_labels = support_labels.long().to(device)
        query_images = query_images.float().to(device)
        query_labels = query_labels.long().to(device)

        if args.algorithm == 'MAML':
            loss, acc = maml_loop(model, support_images, support_labels, query_images, query_labels,
                                  args.test_inner_step, 0, is_train=False)
        if args.algorithm == 'Meta-SGD':
            loss, acc = meta_sgd_loop(model, support_images, support_labels, query_images, query_labels,
                                  args.test_inner_step, 0, 0, meta_sgd=0, is_train=False)
        if args.algorithm == 'Mada-SGD':
            loss, acc, predict = mada_sgd_loop(model, support_images, support_labels, query_images, query_labels,
                                     args.test_inner_step, 0, 0, 0, mada_sgd, is_train=False)
            val_predict.append(predict)
        # Must use .item()  to add total loss, or will occur GPU memory leak.
        # Because dynamic graph is created during forward, collect in backward.
        val_loss.append(loss.item())
        val_acc.append(acc)
        val_label.append(query_labels.data.cpu().numpy())
    
    val_label = np.array(val_label).flatten()
    val_predict = np.array(val_predict).flatten()
    matrix = confusion_matrix(val_label, val_predict)
    print('############ confusion matrix ########### \n', matrix / args.test_episode_num * 287 / 286)
        
    print("=> val_loss: {:.4f}   val_acc: {:.4f}".
          format(np.mean(val_loss), np.mean(val_acc)))
    print("=> Min_Acc: {:.4f}   Max_acc: {:.4f}   Mean_Acc: {:.4f}   Std: {:.6f}".
          format(np.min(val_acc), np.max(val_acc), np.mean(val_acc), np.std(val_acc)))
    
    
