'''
    -------*- coding: utf-8 -*-------
    -------train loop------
    -------------meta-sgd_train---------------
'''

import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import numpy as np
from args import args


def maml_loop(model, support_images, support_labels, query_images, query_labels, inner_step, model_optimizer, is_train=True):
    """
    Train the model using MAML algorithm.
    Args:
        model: Any model
        support_images: several task support images
        support_labels: several  support labels
        query_images: several query images
        query_labels: several query labels
        inner_step: support data training step
        meta_sgd: meta sgd optimizer
        model_optimizer: model optimizer
        alpha_optimizer: meta sgd optimizer
        is_train: whether train

    Returns: meta loss, meta accuracy

    """
    meta_loss = []
    meta_acc = []

    #  outer loop 
    for support_image, support_label, query_image, query_label in zip(support_images, support_labels, query_images, query_labels):

        fast_weights = collections.OrderedDict(model.named_parameters())   # 按照顺序对参数进行保存。
        lr = 0.01
        for _ in range(inner_step):      # inner loop
            # Update weight
            support_logit = model.functional_forward(support_image, fast_weights)
            support_loss = nn.CrossEntropyLoss().cuda()(support_logit, support_label)
            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
                       
            meta_weights = []
            for i, ((name, param), grad) in enumerate(zip(fast_weights.items(), grads)):
                meta_weights.append((name, param -  lr * grad))

            fast_weights = collections.OrderedDict(meta_weights)        
            


        # Use trained weight to get query loss            
        query_logit = model.functional_forward(query_image, fast_weights)
        query_prediction = torch.max(query_logit, dim=1)[1]

        query_loss = nn.CrossEntropyLoss().cuda()(query_logit, query_label)
        query_acc = torch.eq(query_label, query_prediction).sum() / len(query_label)

        meta_loss.append(query_loss)
        meta_acc.append(query_acc.data.cpu().numpy())

    # Zero the gradient
    model_optimizer.zero_grad()
    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)

    if is_train:
        torch.autograd.set_detect_anomaly(True)
        meta_loss.backward()
        model_optimizer.step()

    return meta_loss, meta_acc