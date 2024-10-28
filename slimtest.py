import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from os.path import join
from datetime import datetime
# from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, build_candidates_set
# from daisy.model.SLiMRecommender import SLIM
from scipy import sparse
sys.path.append("../")

import scipy.sparse as sp
from sklearn.linear_model import ElasticNet

# TODO this recommender must change to multiprocessing mode and compress into a more beautiful way


class SLIM(object):
    def __init__(self, user_num, item_num, topk=100,
                 l1_ratio=0.1, alpha=1.0, positive_only=True,mode='origin'):
        """
        SLIM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        topk : int, Top-K number
        l1_ratio : float, The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`
        alpha : float, Constant that multiplies the penalty terms
        positive_only : bool, When set to True, forces the coefficients to be positive
        """
        self.md = ElasticNet(alpha=alpha, 
                             l1_ratio=l1_ratio, 
                             positive=positive_only, 
                             fit_intercept=False,
                             copy_X=False,
                             precompute=True,
                             selection='random',
                             max_iter=100,
                             tol=1e0
                            )
        self.item_num = item_num
        self.user_num = user_num
        self.topk = topk

        self.w_sparse = None
        self.A_tilde = None
        self.mode=mode
        print(f'user num: {user_num}, item num: {item_num}')

    def fit(self, df, verbose=True):
        
        name='movielens'
        origin=True
        if self.mode == 'origin':
            train = self._convert_df(self.user_num, self.item_num, df)
        else:
            train=sp.load_npz('./dok_matrix_'+name+'.npz').tocsc()
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for currentItem in range(self.item_num):
            y = train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = train.indptr[currentItem]
            end_pos = train.indptr[currentItem + 1]

            current_item_data_backup = train.data[start_pos: end_pos].copy()
            train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.md.fit(train, y)

            nonzero_model_coef_index = self.md.sparse_coef_.indices
            nonzero_model_coef_value = self.md.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value)-1, self.topk)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):
                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            train.data[start_pos:end_pos] = current_item_data_backup

            if verbose and (time.time() - start_time_printBatch > 300 or (currentItem + 1) % 1000 == 0 or currentItem == self.item_num - 1):
                print('{}: Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}'.format(
                     'SLIMElasticNetRecommender',
                     currentItem+1,
                     100.0* float(currentItem+1)/self.item_num,
                     (time.time()-start_time)/60,
                     float(currentItem)/(time.time()-start_time)))

                sys.stdout.flush()
                sys.stderr.flush()
                                    
                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.w_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                      shape=(self.item_num, self.item_num), dtype=np.float32)

        train = train.tocsr()
        self.A_tilde = train.dot(self.w_sparse).A

        sp.save_npz('sparse_'+name+'_'+self.mode+'_Sparse.npz', self.w_sparse)
        print("save done!")
    def predict(self, u, i):

        return self.A_tilde[u, i]

    def _convert_df(self, user_num, item_num, df):
        """Process Data to make WRMF available"""
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])
        mat = sp.csc_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Baseline')
    parser.add_argument('--dataset', type=str, default='Gowalla')
    parser.add_argument('--topk', type=str, default='[20, 50]')
    parser.add_argument('--l1', type=float, default=1e-6)
    parser.add_argument('--alpha', type=float, default=1e-2)
    parser.add_argument('--mode', type=str, default='origin')
    args = parser.parse_args()

    print(args)

    # load data
    # TRAIN_PATH = join('/media/xreco/RUC/hanze/Slimtest/slim2/daisyRec/data/' + args.dataset + '/' + args.dataset.lower() + '_x0', 'train.txt')
    # TEST_PATH = join('/media/xreco/RUC/hanze/Slimtest/slim2/daisyRec/data/' + args.dataset + '/' + args.dataset.lower() + '_x0', 'test.txt')
    TRAIN_PATH = join('./data/' + args.dataset +'_m1', 'train.txt')
    TEST_PATH = join('./data/' + args.dataset +'_m1', 'test.txt')

    # item popularities
    with open(TRAIN_PATH, 'r') as f:
        train_data = f.readlines()
    
    train_set = {'user':[], 'item':[], 'rating':[]}
    user_num, item_num = 0, 0
    item_set = set()
    interacted_items = {}
    for i, line in enumerate(train_data):
        line = line.strip().split(' ')
        user = int(line[0])
        interacted_items[user] = set()
        for iid in line[1:]:
            iid = int(iid)
            train_set['user'].append(user)
            train_set['item'].append(iid)
            train_set['rating'].append(1.0)

            if iid not in item_set:
                item_num += 1
                item_set.add(iid)
            interacted_items[user].add(iid)

        user_num += 1
        
    # user_num+=1
    # item_num+=1
    train_set = pd.DataFrame(train_set)
    model = SLIM(user_num, item_num, l1_ratio=args.l1, alpha=args.alpha,mode=args.mode)

    print('model fitting...')
    model.fit(train_set)

    print('Finished')