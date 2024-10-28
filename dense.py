# =========================================================================
# Copyright (C) 2020-2023. The UltraGCN Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: This program bundles some third-party utility functions (hit, ndcg, 
# RecallPrecision_ATk, MRRatK_r, NDCGatK_r, test_one_batch, getLabel) under
# the MIT License.
#
# Copyright (C) 2020 Xiang Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def data_param_prepare(config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    params = {}

    embedding_dim = config.getint('Model', 'embedding_dim')
    params['embedding_dim'] = embedding_dim
    ii_neighbor_num = config.getint('Model', 'ii_neighbor_num')
    params['ii_neighbor_num'] = ii_neighbor_num
    model_save_path = config['Model']['model_save_path']
    params['model_save_path'] = model_save_path
    max_epoch = config.getint('Model', 'max_epoch')
    params['max_epoch'] = max_epoch

    params['enable_tensorboard'] = config.getboolean('Model', 'enable_tensorboard')
    
    initial_weight = config.getfloat('Model', 'initial_weight')
    params['initial_weight'] = initial_weight

    dataset = config['Training']['dataset']
    params['dataset'] = dataset
    train_file_path = config['Training']['train_file_path']
    gpu = config['Training']['gpu']
    params['gpu'] = gpu
    device = torch.device('cuda:'+ params['gpu'] if torch.cuda.is_available() else "cpu")
    params['device'] = device
    lr = config.getfloat('Training', 'learning_rate')
    params['lr'] = lr
    batch_size = config.getint('Training', 'batch_size')
    params['batch_size'] = batch_size
    early_stop_epoch = config.getint('Training', 'early_stop_epoch')
    params['early_stop_epoch'] = early_stop_epoch
    w1 = config.getfloat('Training', 'w1')
    w2 = config.getfloat('Training', 'w2')
    w3 = config.getfloat('Training', 'w3')
    w4 = config.getfloat('Training', 'w4')
    params['w1'] = w1
    params['w2'] = w2
    params['w3'] = w3
    params['w4'] = w4
    negative_num = config.getint('Training', 'negative_num')
    negative_weight = config.getfloat('Training', 'negative_weight')
    params['negative_num'] = negative_num
    params['negative_weight'] = negative_weight

    gamma = config.getfloat('Training', 'gamma')
    params['gamma'] = gamma
    lambda_ = config.getfloat('Training', 'lambda')
    params['lambda'] = lambda_
    sampling_sift_pos = config.getboolean('Training', 'sampling_sift_pos')
    params['sampling_sift_pos'] = sampling_sift_pos
    
    test_batch_size = config.getint('Testing', 'test_batch_size')
    params['test_batch_size'] = test_batch_size
    topk = config.getint('Testing', 'topk') 
    params['topk'] = topk

    test_file_path = config['Testing']['test_file_path']

    # dataset processing
    train_data, test_data, train_mat, user_num, item_num, constraint_mat = load_data(train_file_path, test_file_path)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle = True, num_workers=5)
    test_loader = data.DataLoader(list(range(user_num)), batch_size=test_batch_size, shuffle=False, num_workers=5)

    params['user_num'] = user_num
    params['item_num'] = item_num

    # mask matrix for testing to accelarate testing speed
    mask = torch.zeros(user_num, item_num)
    interacted_items = [[] for _ in range(user_num)]
    for (u, i,_) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)


    # Compute \Omega to extend UltraGCN to the item-item co-occurrence graph
    ii_cons_mat_path = './' + dataset + '_ii_constraint_mat'
    ii_neigh_mat_path = './' + dataset + '_ii_neighbor_mat'
    
    # if os.path.exists(ii_cons_mat_path):
    #     ii_constraint_mat = pload(ii_cons_mat_path)
    #     ii_neighbor_mat = pload(ii_neigh_mat_path)
    # else:
    ii_neighbor_mat, ii_constraint_mat,item_mat = get_ii_constraint_mat(train_mat, ii_neighbor_num)


    users = [pair[0] for pair in train_data]
    items = [pair[1] for pair in train_data]

    # 计算用户和物品的度数
    user_degree = np.bincount(users)
    item_degree = np.bincount(items)

    # 假设 res_mat 是一个包含用户索引的 NumPy 数组
    # for user_id, item_indices in enumerate(ii_neighbor_mat):
    #     for item_index in item_indices:
    #         # 将每个用户和项目的索引组合成一个元组，并添加到列表中
    #         # if user_degree[user_id] <20:
    #         train_data.append([user_id, item_index.item(),1])
    for user_id, item_indices in enumerate(item_mat):
        for item_index in item_indices:
            # 将每个用户和项目的索引组合成一个元组，并添加到列表中
            # if item_degree[user_id] <20:
            train_data.append([item_index.item(),user_id,1])
    # unique_pairs_set = set(map(tuple, train_data))
    # train_data=list(map(list, unique_pairs_set))
    # print("train data lens is :  ",len(train_data))
    # train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle = True, num_workers=5)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle = False, num_workers=5)
    # pstore(ii_neighbor_mat, ii_neigh_mat_path)
    # pstore(ii_constraint_mat, ii_cons_mat_path)

    return params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items

 
def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    print('Computing \\Omega for the item-item graph... ')
    data_name='movielens'
    read_path="/media/xreco/RUC/hanze/UltraGCN2/sparse_"+data_name+"_origin_Sparse.npz"
    loaded_matrix = sp.load_npz(read_path)
    print(loaded_matrix.shape,train_mat.shape)
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    n_users=train_mat.shape[0]

    num_neighbors2=5
    res_mat = torch.zeros((n_users, num_neighbors))
    res_sim_mat = torch.zeros((n_users, num_neighbors))
    res_mat2 = torch.zeros((n_items, num_neighbors2))
    res_sim_mat2 = torch.zeros((n_items, num_neighbors2))
    # if ii_diagonal_zero:
    #     A[range(n_items), range(n_items)] = 0
    A=loaded_matrix
    B=train_mat.dot(loaded_matrix)
    
    # items_D = np.sum(A, axis = 0).reshape(-1)+36
    # users_D = np.sum(A, axis = 1).reshape(-1)+36

    # # u_D=np.sqrt(np.sum(users_D)/len(users_D))
    # # i_D=np.sqrt(np.sum(items_D)/len(items_D))
    # # print(u_D,i_D)
    # beta_uD = (6 / np.sqrt(users_D)).reshape(-1, 1)
    # # beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    # beta_iD = (6 / np.sqrt(items_D)).reshape(1, -1)
    # all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    # for i in range(n_items):
    #     row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
    #     row_sims, row_idxs = torch.topk(row, num_neighbors)
    #     res_mat[i] = row_idxs
    #     res_sim_mat[i] = row_sims               
    #     if i % 15000 == 0:
    #         print('i-i constraint matrix {} ok'.format(i))
    num_users = B.shape[0]
   
    # 遍历每个用户，提取索引和值
    # all_ii_constraint_mat = beta_uD.dot(beta_iD)
    for user_id in range(B.shape[0]):
        # 获取当前用户的所有项目及其值o
        # print(type(loaded_matrix.getrow(user_id).toarray().ravel()),type(all_ii_constraint_mat[user_id].ravel()))
        # print(loaded_matrix.getrow(user_id).toarray().ravel().shape,all_ii_constraint_mat[user_id].ravel().shape)
        # user_row = loaded_matrix.getrow(user_id).toarray()*((np.asarray(all_ii_constraint_mat[user_id])).ravel())
        user_row = B.getrow(user_id).toarray().ravel()
        # 找到最大的K个值及其索引
        # print(user_row.shape)
        top_k_indices = np.argsort(user_row)[-num_neighbors:]
        top_k_values = user_row[top_k_indices]

        # 添加到列表
        res_mat[user_id] = torch.tensor(top_k_indices)
        res_sim_mat[user_id] = torch.tensor(top_k_values)
    print(type(B))
    B2=B.tocsc()
    print(B2.shape)
    for item_id in range(B2.shape[1]):
        # 获取当前用户的所有项目及其值o
        # print(type(loaded_matrix.getrow(user_id).toarray().ravel()),type(all_ii_constraint_mat[user_id].ravel()))
        # print(loaded_matrix.getrow(user_id).toarray().ravel().shape,all_ii_constraint_mat[user_id].ravel().shape)
        # user_row = loaded_matrix.getrow(user_id).toarray()*((np.asarray(all_ii_constraint_mat[user_id])).ravel())
        item_row = B2.getcol(item_id).toarray().ravel()
        # 找到最大的K个值及其索引
        # print(user_row.shape)
        top_k_indices = np.argsort(item_row)[-num_neighbors2:]
        top_k_values =item_row[top_k_indices]

        # 添加到列表
        res_mat2[item_id] = torch.tensor(top_k_indices)
        res_sim_mat2[item_id] = torch.tensor(top_k_values)
    # print(user_item_pairs)


    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float(),res_mat2.long()

    
def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []

    n_user += 1
    m_item += 1

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i],1])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # construct degree matrix for graphmf

    items_D = np.sum(train_mat, axis = 0).reshape(-1)+0
    users_D = np.sum(train_mat, axis = 1).reshape(-1)+0

    # beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_uD = (1 / np.sqrt(users_D + 1)).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}

    return train_data, test_data, train_mat, n_user, m_item, constraint_mat


def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))


def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos):
	neg_candidates = np.arange(item_num)

	if sampling_sift_pos:
		neg_items = []
		for u in pos_train_data[0]:
			probs = np.ones(item_num)
			probs[interacted_items[u]] = 0
			probs /= np.sum(probs)

			u_neg_items = np.random.choice(neg_candidates, size = neg_ratio, p = probs, replace = True).reshape(1, -1)
	
			neg_items.append(u_neg_items)

		neg_items = np.concatenate(neg_items, axis = 0) 
	else:
		neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), neg_ratio), replace = True)
	
	neg_items = torch.from_numpy(neg_items)
	
	return pos_train_data[0], pos_train_data[1], neg_items	# users, pos_items, neg_items


class UltraGCN(nn.Module):
    def __init__(self, params, constraint_mat, ii_constraint_mat, ii_neighbor_mat):
        super(UltraGCN, self).__init__()
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.embedding_dim = params['embedding_dim']
        self.w1 = params['w1']
        self.w2 = params['w2']
        self.w3 = params['w3']
        self.w4 = params['w4']

        self.negative_weight = params['negative_weight']
        self.gamma = params['gamma']
        self.lambda_ = params['lambda']

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat

        self.initial_weight = params['initial_weight']
        self.initial_weights()

        self.x1=params['item_num']
        self.x2=params['negative_num']
        self.x3=params['sampling_sift_pos']
        self.device=params['device']

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()                                                                                                                                                                                                                                                         
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(device)
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        
        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)), self.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight,weight):
        device = self.get_device()
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')
        # print(weight)
        loss = weight*(pos_loss + neg_loss * self.negative_weight)
      
        return loss.sum()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items].to(device))    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(device)     # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)
        
        # print(users.shape,pos_items.shape)
        # user_expanded=users.repeat_interleave(20)

        # user_item_pairs = [user_expanded, self.ii_neighbor_mat[pos_items].to(device).view(-1)]
        # users2, pos_items, neg_items = Sampling(user_item_pairs, self.x1, 1, 0, self.x3)
        # loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
        # print(users2.shape,pos_items.shape,neg_items.shape)
        loss1 = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        # user_embeds2 = self.user_embeds(users2).view(-1,10,64)
        # item_embeds2 = self.item_embeds(neg_items.squeeze().to(device)).view(-1,10,64)
        # user_embeds2 = self.user_embeds(users2).view(-1,20,128)
        # item_embeds2 = self.item_embeds(neg_items.squeeze().to(device)).view(-1,20,128)
        # print(user_embeds2.shape,item_embeds2.shape)
        # loss2= -sim_scores * (user_embeds2 * item_embeds2).sum(dim=-1).sigmoid().log()
        # sim_scores2=self.ii_constraint_mat[neg_items].to(device)
        # print(sim_scores2[:100][:1])
        # loss2=  (1-(user_embeds2 * item_embeds2).sum(dim=-1).sigmoid()).log()
        # loss = loss.sum(-1)
        # print(user_embeds2.shape,item_embeds2.shape,loss2.shape,loss1.shape)
        # loss=loss1+loss2
        loss=loss1
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items,weight):
        omega_weight = self.get_omegas(users, pos_items, neg_items)
        
        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight,weight)
        loss += self.gamma * self.norm_loss()
        # loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def test_foward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
         
        return user_embeds.mm(item_embeds.t())

    def get_device(self):
        return self.user_embeds.weight.device


########################### TRAINING #####################################

def train(model, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params): 
    device = params['device']
    best_epoch, best_recall, best_ndcg = 0, 0, 0
    early_stop_count = 0
    early_stop = False

    batches = len(train_loader.dataset) // params['batch_size']
    if len(train_loader.dataset) % params['batch_size'] != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))
    
    if params['enable_tensorboard']:
        writer = SummaryWriter()
    epoch_to_plot = [1, 20]
    all_item_l2_norms = {epoch: None for epoch in epoch_to_plot}
    for epoch in range(params['max_epoch']):
        model.train() 
        start_time = time.time()
        i=0
        
        for batch, xy in enumerate(train_loader): # x: tensor:[users, pos_items]
            # print((xy[0].shape))
            
            x=xy[:-1]
            users, pos_items, neg_items = Sampling(x, params['item_num'], params['negative_num'], interacted_items, params['sampling_sift_pos'])
            users = users.to(device)
            # print(type(x),len(x))
            # print(users.shape)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            weight=xy[-1].to(device)
            model.zero_grad()
            loss = model(users, pos_items, neg_items,weight)
            # print(loss.shape)
            if params['enable_tensorboard']:
                writer.add_scalar("Loss/train_batch", loss, batches * epoch + batch)
            loss.backward()
            optimizer.step()

            if i==0:
                item_gradients = model.item_embedding.weight.grad
                item_l2_norms = torch.norm(item_gradients, dim=1)
                if epoch in epoch_to_plot:
                    all_item_l2_norms[epoch] = item_l2_norms.detach().cpu().numpy()
                    l2_norms = all_item_l2_norms[epoch]
                    l2_norms_normalized = (l2_norms - l2_norms.min()) / (l2_norms.max() - l2_norms.min())
                    
                    plt.figure()
                    plt.hist(l2_norms_normalized, bins=30, alpha=0.75)
                    plt.title(f'Epoch {epoch} - Normalized L2 Norms of Gradients')
                    plt.xlabel('Normalized L2 Norm')
                    plt.ylabel('Frequency')
                    plt.show()
                    plt.save(str(epoch)+".png")
    
            i+=1
        
        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        if params['enable_tensorboard']:
            writer.add_scalar("Loss/train_epoch", loss, epoch)

        need_test = True
        if epoch < 50 and epoch % 5 != 0:
            need_test = False
            
        if need_test:
            start_time = time.time()
            F1_score, Precision, Recall, NDCG = test(model, test_loader, test_ground_truth_list, mask, params['topk'], params['user_num'])
            if params['enable_tensorboard']:
                writer.add_scalar('Results/recall@20', Recall, epoch)
                writer.add_scalar('Results/ndcg@20', NDCG, epoch)
            test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
            
            print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
            print("Loss = {:.5f}, F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(loss.item(), F1_score, Precision, Recall, NDCG))

            if Recall > best_recall:
                best_recall, best_ndcg, best_epoch = Recall, NDCG, epoch
                early_stop_count = 0
                torch.save(model.state_dict(), params['model_save_path'])

            else:
                early_stop_count += 1
                if early_stop_count == params['early_stop_epoch']:
                    early_stop = True
        
        if early_stop:
            print('##########################################')
            print('Early stop is triggered at {} epochs.'.format(epoch))
            print('Results:')
            print('best epoch = {}, best recall = {}, best ndcg = {}'.format(best_epoch, best_recall, best_ndcg))
            print('The best model is saved at {}'.format(params['model_save_path']))
            break

    writer.flush()

    print('Training end!')


########################### TESTING #####################################

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def RecallPrecision_ATk(test_data, r, k):
	"""
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
	right_pred = r[:, :k].sum(1)
	precis_n = k
	
	recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
	recall_n = np.where(recall_n != 0, recall_n, 1)
	recall = np.sum(right_pred / recall_n)
	precis = np.sum(right_pred) / precis_n
	return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
	"""
    Mean Reciprocal Rank
    """
	pred_data = r[:, :k]
	scores = np.log2(1. / np.arange(1, k + 1))
	pred_data = pred_data / scores
	pred_data = pred_data.sum(1)
	return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
	"""
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
	assert len(r) == len(test_data)
	pred_data = r[:, :k]

	test_matrix = np.zeros((len(pred_data), k))
	for i, items in enumerate(test_data):
		length = k if k <= len(items) else len(items)
		test_matrix[i, :length] = 1
	max_r = test_matrix
	idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
	dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
	dcg = np.sum(dcg, axis=1)
	idcg[idcg == 0.] = 1.
	ndcg = dcg / idcg
	ndcg[np.isnan(ndcg)] = 0.
	return np.sum(ndcg)


def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test(model, test_loader, test_ground_truth_list, mask, topk, n_user):
    users_list = []
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        model.eval()
        for idx, batch_users in enumerate(test_loader):
            
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users) 
            rating = rating.cpu()
            rating += mask[batch_users]
            
            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users])

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg
        
    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config file path')
    args = parser.parse_args()

    print('###################### Dense ######################')

    print('Loading Configuration...')
    params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items = data_param_prepare(args.config_file)
    
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)

    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    ultragcn = ultragcn.to(params['device'])
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])

    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)

    print('END')
