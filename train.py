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
from algorithms.mada_sgd import adaptiveSGD, adaptiveSGD_V2, adaptiveSGD2, adaptiveSGD3, mada_sgd_loop
import copy
import scipy.io

from data.dataset.MiniImagenet import MiniImagenet

if __name__ == '__main__':   
    
    if args.adaptiveSGD:
        args.test_inner_step = args.train_inner_step
   
    #   backbone networks    
    model = get_model(args, device)
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
    train_dataset, test_dataset = get_dataset_miniimagenet(args)
    
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)      
    
    print('-' * 100)
    print('train image num: {}, train image class: {}, test image num: {}, test image class: {}'.format(len([img for imgs in train_dataset.data for img in imgs]), train_dataset.cls_num,
                                                                                                        len([img for imgs in test_dataset.data for img in imgs]), test_dataset.cls_num))

    print('-' * 100)
    
    if args.algorithm == 'MAML':
        model_params = [p for p in model.parameters() if p.requires_grad]
        model_optimizer = optim.Adam(model_params, lr=args.lamda)        
            
    if args.algorithm == 'Meta-SGD':    
        meta_sgd = MetaSGD(copy.deepcopy(model))   # 对参数进行深度复制，与原参数储存下不同的内存中。
        model_params = [p for p in model.parameters() if p.requires_grad]
        alpha_params = [p for p in meta_sgd.model.parameters() if p.requires_grad]

        model_optimizer = optim.Adam(model_params, lr=1e-3)
        alpha_optimizer = optim.Adam(alpha_params, lr=1e-4)
    if args.algorithm == 'Mada-SGD':                       
        #-------model-level + layer-level------------meda_sgd-------------------
        mada_sgd = adaptiveSGD(inner_step=args.train_inner_step, layer_num=len(list(model.parameters())))
        mada_sgd = adaptiveSGD_V2(inner_step=args.train_inner_step, merge_step=args.merge_inner_step, layer_num=len(list(model.parameters())))
        beta_params = [mada_sgd.beta]
        eta_params = [mada_sgd.eta]
        
        # -------element-level-----mada_sgd--------        
        # beta_models = [copy.deepcopy(model) for i in range(args.train_inner_step)]   # element-level
        # eta_models = [copy.deepcopy(model) for i in range(args.train_inner_step)]
        # mada_sgd = adaptiveSGD3(beta_models=beta_models, eta_models=eta_models,
                                # inner_step=args.train_inner_step, layer_num=len(list(model.parameters())))

        # beta_params = [p for model in mada_sgd.beta_models for p in model.parameters() if p.requires_grad]
        # eta_params = [p for model in mada_sgd.eta_models for p in model.parameters() if p.requires_grad]
        # beta_eta_params = beta_params + eta_params                         
        model_params = [p for p in model.parameters() if p.requires_grad]                                  
        
        model_optimizer = optim.Adam(model_params, lr=9e-4)      
        beta_optimizer = optim.Adam(beta_params, lr=2e-3)
        eta_optimizer = optim.Adam(eta_params, lr=2e-3)
        
        print('model optimizer lr: {},   beta optimizer lr: {},   eta optimizer lr: {}'.format(9e-4, 2e-3, 2e-3))
        print('-' * 100)

    best_acc = 0
    val_acc_curve = []
    val_loss_curve = []
    beta = []
    eta = []

    for epoch in range(args.epochs):
        # model.train()
        # ada_sgd.beta_model.train()       

        train_bar = tqdm(train_loader)
        for batch, (support_images, support_labels, query_images, query_labels) in enumerate(train_bar):
            model.train()
            train_bar.set_description("epoch {}/{}".format(epoch + 1, args.epochs))

            # save beta & eta
            aa = copy.deepcopy(mada_sgd.beta.data).reshape(1, -1)
            bb = copy.deepcopy(mada_sgd.eta.data).reshape(1, -1)
            beta.append(aa)
            eta.append(bb)
            
            train_acc = []
            val_acc = []
            train_loss = []
            val_loss = []
            val_predict = []
            val_label = []
            # Get variables
            support_images = support_images.float().to(device)
            support_labels = support_labels.long().to(device)
            query_images = query_images.float().to(device)
            query_labels = query_labels.long().to(device)

            if args.algorithm == 'MAML':
                loss, acc = maml_loop(model, support_images, support_labels, query_images, query_labels,
                                      args.train_inner_step, model_optimizer, is_train=True)
            if args.algorithm == 'Meta-SGD':
                loss, acc = meta_sgd_loop(model, support_images, support_labels, query_images, query_labels,
                                      args.train_inner_step, model_optimizer, alpha_optimizer, meta_sgd, is_train=True)
            if args.algorithm == 'Mada-SGD':
                loss, acc, _ = mada_sgd_loop(model, support_images, support_labels, query_images, query_labels,
                                      args.train_inner_step, model_optimizer, beta_optimizer, eta_optimizer, mada_sgd, is_train=True)
                                    
            train_loss.append(loss.item())
            train_acc.append(acc)
            train_bar.set_postfix(loss="{:.4f}".format(loss.item()))
            
            #  每5个batch测试一次
            if (batch+1) % 5 == 0:
                model.eval()  
                # ada_sgd.beta_model.eval()
                for support_images, support_labels, query_images, query_labels in test_loader:

                    # Get variables
                    support_images = support_images.float().to(device)
                    support_labels = support_labels.long().to(device)
                    query_images = query_images.float().to(device)
                    query_labels = query_labels.long().to(device)

                    if args.algorithm == 'MAML':
                        loss, acc = maml_loop(model, support_images, support_labels, query_images, query_labels,
                                              args.test_inner_step, model_optimizer, is_train=False)
                    if args.algorithm == 'Meta-SGD':
                        loss, acc = meta_sgd_loop(model, support_images, support_labels, query_images, query_labels,
                                              args.test_inner_step, model_optimizer, alpha_optimizer, meta_sgd, is_train=False)
                    if args.algorithm == 'Mada-SGD':
                        loss, acc, predict = mada_sgd_loop(model, support_images, support_labels, query_images, query_labels,
                                                 args.test_inner_step, model_optimizer, beta_optimizer, eta_optimizer, mada_sgd, is_train=False)
                        val_predict.append(predict)
                    # Must use .item()  to add total loss, or will occur GPU memory leak.
                    # Because dynamic graph is created during forward, collect in backward.
                    val_loss.append(loss.item())
                    val_acc.append(acc)
                    val_label.append(query_labels)
                if args.adaptiveSGD:
                    # print('inner step weight factors: {}'.format(mada_sgd.beta))
                    # print('inner step learning factors: {}'.format(mada_sgd.eta))
                    # beta = [beta.detach().cpu() for weight in ada_sgd.beta for beta in weight]
                    # for w in beta:
                    #     np.savetxt('./results/beta.txt', np.array(w).reshape(1, -1))
                    pass
                                                  
                print("\n=> train_loss: {:.4f}   train_acc: {:.4f}   val_loss: {:.4f}   val_acc: {:.4f}".
                      format(np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc)))
                print("=> Min_Acc: {:.4f}   Max_acc: {:.4f}   Mean_Acc: {:.4f}   Std: {:.6f}".
                      format(np.min(val_acc), np.max(val_acc), np.mean(val_acc), np.std(val_acc)))
                
                val_acc_curve.append(np.mean(val_acc))
                val_loss_curve.append(np.mean(val_loss))                  
                
                if np.mean(val_acc) > best_acc:
                    best_acc = np.mean(val_acc)
                    save_path = ('./results/models/' + args.dataset  + '-' + args.backbone + '-' + args.algorithm 
                                 + '-' + str(args.n_way) + 'way' + '-' + str(args.k_shot) + 'shot' + '-' +
                                 'epoch-' + str(epoch+1) + '-' + 'batch-'+ str(batch+1) + '-' + str('%0.4f' % best_acc) + 'acc.pt')
                    if args.algorithm == 'MAML':
                        torch.save(model, save_path)
                    if args.algorithm == 'Meta-SGD':
                        torch.save({'base_learner': model,
                                    'alpha': meta_sgd}, save_path)
                    if args.algorithm == 'Mada-SGD':    
                        torch.save({'base_learner': model,
                                'algorithm': mada_sgd,
                                'val_acc': val_acc,
                                'val_label': val_label,
                                'val_predict': val_predict}, save_path)

    beta = torch.cat(beta, dim=0)
    eta = torch.cat(eta, dim=0)
    scipy.io.savemat('./results/models/beta_eta.mat', {'beta': beta.data.cpu().numpy(), 'eta': eta.data.cpu().numpy()})
    scipy.io.savemat('./results/models/val_acc_loss_curve.mat', {'val_acc':val_acc_curve, 'val_loss':val_loss_curve})
    
