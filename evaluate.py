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
from tqdm import tqdm

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
    

    test_batch_size = config.getint('Testing', 'test_batch_size')
    params['test_batch_size'] = test_batch_size
    topk = config.getint('Testing', 'topk') 
    params['topk'] = topk

    test_file_path = config['Testing']['test_file_path']
    # params['modes']= config['Testing']['modes']
    # dataset processing
    train_data, test_data, train_mat, user_num, item_num= load_data(train_file_path, test_file_path)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle = True, num_workers=5)
    test_loader = data.DataLoader(list(range(user_num)), batch_size=test_batch_size, shuffle=False, num_workers=5)

    params['user_num'] = user_num
    params['item_num'] = item_num

   
    # mask matrix for testing to accelarate testing speed
    mask = torch.zeros(user_num, item_num)
    for (u, i) in train_data:
        mask[u][i] = -np.inf

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    return params, test_loader, mask, test_ground_truth_list, train_mat




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
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)
    
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
   
    # construct degree matrix for graphmf

    non_zero_elements_count = len(train_mat)

    

    return train_data, test_data, train_mat, n_user, m_item


class SD(nn.Module):
    def __init__(self, params,train_mat):
        super(SD, self).__init__()
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        
        self.train_mat=train_mat

        # self.alpha=5
        paths= "./sad_" + params['dataset']+".pt"
        x=torch.load(paths)
        user_embed=x['user_embeds.weight']
        item_embed=x['item_embeds.weight']
        print(user_embed.shape,item_embed.shape)


        names=params['dataset']
        if names=="yelp2018":
            names="yelp"
        elif names=="amazon": 
            names="amazonbooks"
        elif names=="ml-1m":
            names="movielens"
        elif names=="electronics":
            names="amazonelectronics"
        elif names=="amazonmovies":
            names="amazonmovies"
        elif names=="amazonbeauty":
            names="amazonbeauty"
        # print("hello,T: ",T.shape,type(T))
        self.user_embeds = nn.Embedding.from_pretrained(torch.nn.Parameter(user_embed))
        self.item_embeds = nn.Embedding.from_pretrained(torch.nn.Parameter(item_embed))
        #计算得到U-I的Dense Matrix，根据规则对U-I rating矩阵做微调


        mode=params['modes']
        selections=['origin','noS2D','ablation']
        if mode=="train":
            ms=0
        else:
            ms=2
        self.read_path="./sparse_"+names+"_"+selections[ms]+"_Sparse.npz"
        loaded_matrix = sp.load_npz(self.read_path)
        P=user_embed.cpu()@item_embed.T.cpu()  

        if mode == "train":
            if names=="gowalla":
                K1,K2=5,10
                # K1,K2=0,0
            elif names=="yelp":
                # K1,K2=0,25 #0,25/0.5,(0,30)/0.3
                K1,K2=0,25
            elif names=="amazoncds":
                K1,K2=0,20        
            elif names=="amazonbooks":
                K1,K2=0,10 #2,5
            elif names=="movielens":
                # K1,K2=5,10
                K1,K2=5,25
                # K1,K2=0,0
            elif names=="amazonelectronics":
                K1,K2=0,15 
            elif names=="amazonmovies":
                K1,K2=0,0
            elif names=="amazonbeauty":
                K1,K2=0,0            
        elif mode == "test":
            if names=="gowalla": 
                K1,K2=0,15
                # K1,K2=0,15
                #  K1,K2=0,0
            elif names=="yelp": 
                # K1,K2=0,30
                K1,K2=0,50
            elif names=="amazoncds":
                K1,K2=0,0
            elif names=="amazonbooks":
                K1,K2=0,0
            elif names=="movielens":
                K1,K2=0,0
            elif names=="amazonelectronics":
                K1,K2=0,0
            elif names=="amazonmovies":
                K1,K2=0,0
            elif names=="amazonbeauty":
                K1,K2=0,0
        if names=="gowalla":
            self.beta=10
            # self.beta=10
        elif names=="yelp":
            self.beta=15 #15
        elif names=="amazoncds":
            # self.beta=50
            self.beta=100
        # elif names=="amazonbooks":
        #     self.beta=50  
        elif names=="amazonbooks":
            # self.beta=200
            self.beta=100
        elif names=="movielens":
            self.beta=3
        elif names=="amazonelectronics":
            self.beta=10
        elif names=="amazonmovies":
            self.beta=100
        elif names=="amazonbeauty":
            self.beta=50
        # SPP = csr_matrix(PP)
        # del PP
        print("PPok")
        alpha=0.3
        P2 =retain_top_k_values(P,K1,K2,alpha)#amazon 5,10 yelp 5,10 gowalla:5,10?  yelp 0,5 ,5,5 amazoncds 0,20?
        del P
        print("ok")           
        Q2=train_mat.toarray()
        Q3=torch.tensor(Q2)
        # P2=P2.cuda()
        # Q3=Q3.cuda()
        del Q2 
        Q3[(P2==alpha)&(Q3==0)]=alpha
        Q3[(P2==alpha)&(Q3==1)]+=alpha
        del P2
        print("ok2")
        # Q4=Q3.cpu().numpy()
        Q4=Q3.numpy()
        Q5=sp.dok_matrix(Q4)
        if mode=="train":
            sp.save_npz('dok_matrix_'+names+'.npz', Q5.tocsr())
        else:
            QQ=loaded_matrix
            self.Q6=Q5@QQ
        print("done")

   

    def test_foward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
        user = users.cpu()

        U_1 = np.array(self.Q6[user,:].todense())
        ret= torch.tensor(U_1).cuda()
        

        return self.beta*ret+user_embeds.mm(item_embeds.t()) #gowalla is 10 yelp is 10 amazon is 25 amazoncds is 100  now:yelp is 15
    def get_device(self):
        return self.user_embeds.weight.device


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

def keep_top_k(matrix, k):
    # 遍历每一行
    print(matrix.shape)
    for i in range(matrix.shape[0]):
        if i %3000==0:
            print(i)
        row = matrix[i, :]
        # 获取每行中前 K 个最大值的索引
        top_k_indices = np.argsort(row)[-k:]
        # print(len(set(range(len(row)))),len(set(top_k_indices)))
        # 将除了前 K 个最大值之外的其他元素置为 0
        mask = np.zeros_like(row, dtype=bool)
        mask[top_k_indices] = True
        row[~mask] = 0  # 使用 ~mask 将非最大值位置置为 0


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


    print("F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(F1_score, Precision, Recall, NDCG))



def retain_top_k_values(tensor, k1,k2,alpha=0.5):
    # 获取每个用户最大的K个值的索引
    top_k_indices = torch.topk(tensor, k1, dim=1, largest=True)[1].cpu()
    # 获取每个用户最小的K个值的索引
    # bottom_k_indices = torch.topk(tensor, k, dim=1, largest=False)[1].cpu()
    top_k_indices2 = torch.topk(tensor, k2, dim=0, largest=True)[1].cpu()


    # top_k_indices3 = torch.topk(tensor, 30000, dim=1, largest=False)[1].cpu()
    # top_k_indices4 = torch.topk(tensor, 30000, dim=0, largest=False)[1].cpu()

    # 创建一个新的tensor，初始值为0
    result_tensor = torch.zeros_like(tensor.cpu())
    
    
    result_tensor.scatter_(1, top_k_indices, alpha)
    result_tensor.scatter_(0, top_k_indices2, alpha)
    # result_tensor.scatter_(1, top_k_indices3, 0.6)
    # result_tensor.scatter_(0, top_k_indices4, 0.6)
    # 将最大的K个值对应的索引位置设为1
    # result_tensor.scatter_(1, top_k_indices[:k1//2], 0.5)
    # result_tensor.scatter_(0, top_k_indices2[:k2//2], 0.5)

    # result_tensor.scatter_(1, top_k_indices[k1//2:], 0.5)
    # result_tensor.scatter_(0, top_k_indices2[k2//2:], 0.5)

    # 将最小的K个值对应的索引位置设为-1
    # result_tensor.scatter_(1, bottom_k_indices, -1)

    return result_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config file path')
    parser.add_argument('--modes', type=str, help='config file path')
    args = parser.parse_args()

    params, test_loader, mask, test_ground_truth_list,train_mat = data_param_prepare(args.config_file)
    params['modes']=args.modes
    print(params)
    
    sparse_dense = SD(params,train_mat)
    sparse_dense = sparse_dense.to(params['device'])

    if params['modes'] =='test':
        test(sparse_dense, test_loader, test_ground_truth_list, mask, params['topk'], params['user_num'])

    print('END')



