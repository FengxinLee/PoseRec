#!/usr/bin/env python
# pylint: disable=W0201
import argparse
import numpy as np
import pandas as pd
import torch
from video_side import PoseModel_Disentangle

import pickle
import random
import os


def set_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      # 为当前GPU设置随机种子（只用一块GPU）
        torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。

def get_parser():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--pose_weights', default='./model/best_model_pose.pt', help='the weights for network initialization')
    parser.add_argument('--pose_file', default='./data/sample.pkl', help='the input pose array of video side')
    parser.add_argument('--top_k', default=5, type=int, help='recommend top k list')
    return parser
    

def load_model(weights):
    posemodel = PoseModel_Disentangle(embedding_size=256, num_cat=4, layer=3, edge_importance_weighting=True)
    posemodel.load_state_dict(torch.load(weights))
    posemodel.eval()
    return posemodel
    
def get_data(filename):
    
    video_arrary = pickle.load(open(filename, 'rb'))
    video_arrary = torch.Tensor(video_arrary)
    video_arrary = video_arrary.unsqueeze(4)
    video_arrary = video_arrary.permute(0, 3, 1, 2, 4).contiguous()
    # video_arrary = video_arrary[5:10]

    item_embedding = pickle.load(open('./data/item_feature.pkl', 'rb'))
    for i in range(len(item_embedding)):
        item_embedding[i] = item_embedding[i]/np.linalg.norm(item_embedding[i], axis=1, keepdims=True)
        item_embedding[i] = torch.Tensor(item_embedding[i]) 

    prototype_dis = np.load('./data/prototype_dis.npy', allow_pickle=True)
    prototype_dis = torch.Tensor(prototype_dis)
    item_data = pd.read_csv('./data/item_data.csv', header=None).iloc[:,0].to_list()
 
    return video_arrary, item_embedding, prototype_dis, item_data

if __name__ == "__main__":
    set_seed(1)

    arg = get_parser().parse_args()

    pose_file = arg.pose_file
    pose_weights = arg.pose_weights

    video_arrary, item_embedding, prototype_dis, item_data = get_data(filename=pose_file)
    posemodel = load_model(weights=pose_weights)

    video_feature = posemodel(video_arrary)
    
    rank_score = torch.zeros(video_feature[0].shape[0], item_embedding[0].shape[0])
    for i in range(len(video_feature)):
        video_feature[i] = video_feature[i] / torch.norm(video_feature[i], dim=1, keepdim=True)
        all_score = torch.matmul(video_feature[i], item_embedding[i].T)
        rank_score += torch.mul(prototype_dis[:,i], all_score)

    k = arg.top_k
    top_k = torch.topk(rank_score, k=k, dim=1)[1].cpu().numpy()
    for i in range(rank_score.shape[0]):
        for j in range(k):
            print(item_data[top_k[i, j]], end=' ')
        print()

