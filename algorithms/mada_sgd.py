'''
    -------*- coding: utf-8 -*-------
    -------train loop------
    -------------mada-sgd_train---------------
'''

import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import numpy as np


#   待测试。。。
class adaptiveSGD_V2:
    """
    learnable parameters + model-level
    fast weight update based on mada-sgd
    """
    def __init__(self, inner_step, merge_step, layer_num):
        self.inner_step = inner_step
        self.merge_step = merge_step
        self.layer_num = layer_num
        # 权重因子
        beta = torch.ones(merge_step).cuda()
        self.beta = torch.nn.Parameter(beta, requires_grad=True)
        self.beta.data.fill_(1/merge_step)    #初始化
        # 更新因子
        eta = torch.ones(1).cuda()
        self.eta = torch.nn.Parameter(eta, requires_grad=True)
        self.eta.data.fill_(1e-2)    #初始化
        
    def update_weights(self, step_grads, step_weights):       

        # 对参数进行加权
        fast_weights = []
        for i, ((name, params), grad) in enumerate(zip(step_weights.items(), step_grads)):                        
            
            weight_layer = params - self.eta * grad          
            
            fast_weights.append((name, weight_layer))
        
        fast_weights = collections.OrderedDict(fast_weights)        
        
        return fast_weights
        
    def merge_weights(self, inner_step, weights):
        # 判断weights数量
        self.inner_step = inner_step
        if self.inner_step == 1:
            weight = weights[0]
        else:
            weight = weights[0]

        # 对参数进行加权
        fast_weights = []
        for i, (name, _) in enumerate(weight.items()):
            
            weight_layer = 0
            for j in range(self.merge_step):
                layer = self.beta[self.merge_step-j-1] * weights[self.inner_step-j-1][name]       # 倒序加权求和
                weight_layer += layer
            
            fast_weights.append((name, weight_layer))
        
        fast_weights = collections.OrderedDict(fast_weights)        
        
        return fast_weights



class adaptiveSGD:
    """
    learnable parameters + model-level
    fast weight update based on mada-sgd
    """
    def __init__(self, inner_step, layer_num):
        self.inner_step = inner_step
        self.layer_num = layer_num
        # 权重因子
        beta = torch.ones(inner_step).cuda()
        self.beta = torch.nn.Parameter(beta, requires_grad=True)
        self.beta.data.fill_(1/inner_step)    #初始化
        # 更新因子
        eta = torch.ones(inner_step).cuda()
        self.eta = torch.nn.Parameter(eta, requires_grad=True)
        self.eta.data.fill_(1e-2)    #初始化
        
    def update_weights(self, step, step_grads, step_weights):       

        # 对参数进行加权
        fast_weights = []
        for i, ((name, params), grad) in enumerate(zip(step_weights.items(), step_grads)):                        
            
            weight_layer = self.beta[step] * params - self.eta[step] * grad          
            
            fast_weights.append((name, weight_layer))
        
        fast_weights = collections.OrderedDict(fast_weights)        
        
        return fast_weights
        
    def merge_weights(self, weights):
        # 判断weights数量
        if self.inner_step == 1:
            weight = weights[0]
        else:
            weight = weights[0]

        # 对参数进行加权
        fast_weights = []
        for i, (name, _) in enumerate(weight.items()):
            
            weight_layer = 0
            for j in range(self.inner_step):
                layer = weights[j][name]
                weight_layer += layer
            
            fast_weights.append((name, weight_layer))
        
        fast_weights = collections.OrderedDict(fast_weights)        
        
        return fast_weights
        
class adaptiveSGD2:
    """
    learnable parameters + layer-level
    fast weight update based on mada-sgd
    """
    def __init__(self, inner_step, layer_num):
        self.inner_step = inner_step
        self.layer_num = layer_num
        # 权重因子
        beta = torch.ones(inner_step, layer_num).cuda()
        self.beta = torch.nn.Parameter(beta, requires_grad=True)
        self.beta.data.fill_(1/inner_step)    #初始化
        # 更新因子
        eta = torch.ones(inner_step, layer_num).cuda()
        self.eta = torch.nn.Parameter(eta, requires_grad=True)
        self.eta.data.fill_(1e-2)    #初始化
        
    def update_weights(self, step, step_grads, step_weights):       

        # 对参数进行加权
        fast_weights = []
        for i, ((name, params), grad) in enumerate(zip(step_weights.items(), step_grads)):                        
            
            weight_layer = self.beta[step,i] * params - self.eta[step,i] * grad 
            
            fast_weights.append((name, weight_layer))
        
        fast_weights = collections.OrderedDict(fast_weights)        
        
        return fast_weights
        
    def merge_weights(self, weights):
        # 判断weights数量
        if self.inner_step == 1:
            weight = weights[0]
        else:
            weight = weights[0]

        # 对参数进行加权
        fast_weights = []
        for i, (name, _) in enumerate(weight.items()):
            
            weight_layer = 0
            for j in range(self.inner_step):
                layer = weights[j][name]
                weight_layer += layer
            
            fast_weights.append((name, weight_layer))
        
        fast_weights = collections.OrderedDict(fast_weights)        
        
        return fast_weights
        
class adaptiveSGD3:
    """
    learnable parameters + element-level
    fast weight update based on mada-sgd
    """
    def __init__(self, beta_models, eta_models, inner_step, layer_num):
        self.inner_step = inner_step
        self.layer_num = layer_num
        # 权重因子
        self.beta_models = beta_models
        for model in beta_models:
            weight_init(model, 1/inner_step)      #  参数初始化
        self.beta = [p for model in beta_models for p in model.parameters() if p.requires_grad]
        # 更新因子
        self.eta_models = eta_models
        for model in eta_models:
            weight_init(model, 0.01)       #  参数初始化
        self.eta = [p for model in eta_models for p in model.parameters() if p.requires_grad]
        
    def update_weights(self, step, step_grads, step_weights):       
        # 读取权重因子参数：beta， 更新因子参数：eta
        self.beta = []
        self.eta = []
        for beta_model, eta_model in zip(self.beta_models, self.eta_models):
            beta = []
            eta = []
            for beta_layer, eta_layer in zip(beta_model.parameters(), eta_model.parameters()):
                beta.append(beta_layer)
                eta.append(eta_layer)
            self.beta.append(beta)
            self.eta.append(eta)       
        
        # 对参数进行加权
        fast_weights = []
        for i, ((name, params), grad) in enumerate(zip(step_weights.items(), step_grads)):                        
            
            weight_layer = self.beta[step][i] * params - self.eta[step][i] * grad 
            
            fast_weights.append((name, weight_layer))
        
        fast_weights = collections.OrderedDict(fast_weights)        
        
        return fast_weights
        
    def merge_weights(self, weights):
        # 判断weights数量
        if self.inner_step == 1:
            weight = weights[0]
        else:
            weight = weights[0]

        # 对参数进行加权
        fast_weights = []
        for i, (name, _) in enumerate(weight.items()):
            
            weight_layer = 0
            for j in range(self.inner_step):
                layer = weights[j][name]
                weight_layer += layer
            
            fast_weights.append((name, weight_layer))
        
        fast_weights = collections.OrderedDict(fast_weights)        
        
        return fast_weights       

def weight_init(model, values):
    
    for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.constant(m.weight, values)
                torch.nn.init.constant(m.bias, values)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.constant(m.weight, values)
                torch.nn.init.constant(m.bias, values)
            if isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant(m.weight, values)
                torch.nn.init.constant(m.bias, values)       
        
        
def mada_sgd_loop(model, support_images, support_labels, query_images, query_labels, inner_step, model_optimizer, beta_optimizer, eta_optimizer, mada_sgd, is_train=True):
    """
    Train the model using mada-sgd algorithm.
    Args:
        model: Any model
        support_images: several task support images
        support_labels: several  support labels
        query_images: several query images
        query_labels: several query labels
        inner_step: support data training step
        mada_sgd: mada sgd optimizer
        model_optimizer: model optimizer
        beta_optimizer: mada-sgd optimizer
        eta_optimizer: mada-sgd optimizer
        is_train: whether train

    Returns: meta loss, meta accuracy

    """
    meta_loss = []
    meta_acc = []
    meta_predict = []

    #  outer loop 
    for support_image, support_label, query_image, query_label in zip(support_images, support_labels, query_images, query_labels):

        fast_weights = collections.OrderedDict(model.named_parameters())   # 按照顺序对参数进行保存。

        inner_weights = []
        for step in range(inner_step):      # inner loop
            # Update weight
            support_logit = model.functional_forward(support_image, fast_weights)
            support_loss = nn.CrossEntropyLoss().cuda()(support_logit, support_label)
            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)   # 进行梯度计算，create_graph=True保留当前梯度值，供后续使用。create_graph=False则计算一次后就清空。
            # fast_weights = mada_sgd.update_weights(0, grads, fast_weights)    # 只考虑当前更新步用
            # fast_weights = mada_sgd.update_weights(step, grads, fast_weights)   # 考虑多个更新步用
            fast_weights = mada_sgd.update_weights(grads, fast_weights)    # 调试adaptivesgd_v2用, 2023.08.19.
            inner_weights.append(fast_weights)
            
            
            
        # Use trained weight to get query loss
        # fast_weights = mada_sgd.merge_weights(inner_weights) 
        fast_weights = mada_sgd.merge_weights(inner_step, inner_weights)    # 调试adaptivesgd_v2用, 2023.08.19.
        query_logit = model.functional_forward(query_image, fast_weights)
        query_prediction = torch.max(query_logit, dim=1)[1]

        query_loss = nn.CrossEntropyLoss().cuda()(query_logit, query_label)
        query_acc = torch.eq(query_label, query_prediction).sum() / len(query_label)

        meta_loss.append(query_loss)
        meta_acc.append(query_acc.data.cpu().numpy())
        meta_predict.append(query_prediction.data.cpu().numpy())
       
    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)

    if is_train:                
        # Zero the gradient
        model_optimizer.zero_grad()
        beta_optimizer.zero_grad()
        eta_optimizer.zero_grad()
        
        torch.autograd.set_detect_anomaly(True)       
        meta_loss.backward()
        model_optimizer.step()
        beta_optimizer.step()
        eta_optimizer.step()

    return meta_loss, meta_acc, meta_predict