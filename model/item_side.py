# coding: UTF-8
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import math

class NLPModel_Disentangle(nn.Module):

    def __init__(self, embedding_path, num_cat=1, embedding_size=256, colum_list=None, device=0):
        super(NLPModel_Disentangle, self).__init__()

        self.hidden_size = 768
        self.embedding_size = int(embedding_size//num_cat)
        if colum_list is not None:
            self.num_infor = len(colum_list)
        self.colum_list = colum_list
        self.device = device
        self.num_cat = num_cat

        nlp_embedding = pickle.load(open(embedding_path, 'rb'))
        self.nlp_embedding_list = nn.ModuleList([nn.Embedding.from_pretrained(nlp_embedding[i], freeze=True) for i in range(9)])
        
        self.cat_embedding = nn.Embedding(num_embeddings=self.num_cat, embedding_dim=self.embedding_size)

        self.fc = nn.ModuleList([nn.Linear(self.hidden_size, self.embedding_size) for i in range(self.num_cat)])
        self.attention = nn.ParameterList([nn.Parameter(torch.ones(self.num_infor)) for i in range(self.num_cat)])

        self.fc2 = nn.Linear(self.hidden_size, self.embedding_size)
        self.attention2 = nn.Parameter(torch.ones(self.num_infor))

    def forward(self, idx):
        # get embedding from pretrained embedding
        idx = idx.to(self.device)
        pooled_list = torch.zeros((idx.shape[0], self.num_infor, self.hidden_size), requires_grad=False).to(self.device)
        pooled_list_cat = torch.zeros((idx.shape[0], self.num_infor, self.hidden_size), requires_grad=False).to(self.device)
        
        for i, c in enumerate(self.colum_list):
            pooled_list_cat[:, i, :] = self.nlp_embedding_list[c](idx)*self.attention2[i]
        # get embedding for category embedding
        cat_emb = self.fc2(torch.sum(pooled_list_cat, dim=1))
        category = torch.zeros(idx.shape[0], self.num_cat)

        out = []
        for i_cat in range(self.num_cat):
            for i, c in enumerate(self.colum_list):
                pooled_list[:, i, :] = self.nlp_embedding_list[c](idx)*self.attention[i_cat][i]
          
            # get embedding for each category from different encoder
            feature_emb = self.fc[i_cat](torch.sum(pooled_list, dim=1))
            out.append(feature_emb)
           
            # caculate category
            cat_em = self.cat_embedding(torch.LongTensor([i_cat]).to(self.device)).squeeze(0)
            cat_norm = torch.norm(cat_em)
            category[:, i_cat] = torch.sum(torch.mul(cat_emb, cat_em)/torch.norm(cat_emb)/cat_norm, dim=1)
        category = torch.softmax(category, dim=1)
        return out, category
    
    def get_embedding(self):
        length_em = self.nlp_embedding_list[0].weight.size(0)
        out, category = self.forward(torch.range(length_em))
        return out, category
    
    def get_cat_embedding(self, idx):
        # get embedding from pretrained embedding
        idx = idx.to(self.device)
        pooled_list_cat = torch.zeros((idx.shape[0], self.num_infor, self.hidden_size), requires_grad=False).to(self.device)
        for i, c in enumerate(self.colum_list):
            pooled_list_cat[:, i, :] = self.nlp_embedding_list[c](idx)*self.attention2[i]
        # get embedding for category embedding
        cat_emb = self.fc2(torch.sum(pooled_list_cat, dim=1))
        return cat_emb

    def get_cat_embedding_product(self):
        cat_embedding = self.cat_embedding.weight
        product = torch.zeros(cat_embedding.shape[0], cat_embedding.shape[0])
        for i in range(cat_embedding.shape[0]):
            for j in range(cat_embedding.shape[0]):
                product[i, j] = torch.sum(torch.mul(cat_embedding[i], cat_embedding[j]))
        return torch.sum(product)


