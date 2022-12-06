from numpy import product
import torch
import torch.nn as nn


class ContrastLoss_Disentangle(nn.Module):
    def __init__(self, device):
        super(ContrastLoss_Disentangle, self).__init__()
        self.device = device

    def get_bce_loss(self, p, label):
        p = torch.sigmoid(p)
        return -(torch.log(p) * label + torch.log(1 - p) * (1 - label))
    
    def get_product_matirx(self, pose_features, categories):
        product_matrix = torch.zeros(pose_features[0].size(0), pose_features[0].size(0)).to(self.device)
        for i in range(categories.size(1)):
            for j in range(pose_features[0].size(0)):
                for jj in range(pose_features[0].size(0)):
                    if j > jj:
                        continue
                    if j == jj:
                        product_matrix[j,jj] = 1
                        continue
                    product_matrix[j, jj] += torch.dot(pose_features[i][j], pose_features[i][jj])
                    product_matrix[jj, j] = product_matrix[j, jj]
        return product_matrix

    def forward(self, nlp_features, pose_features, pose_label, nlp_label, nlpid2poseid, poseid2nlpid, categories): 
        loss_norm = []
        self.nlp_features = nlp_features
        self.pose_features = pose_features
        
        # norm loss
        for i in range(len(categories[0])):
            loss_norm_nlp = torch.norm(nlp_features[i], p=2, dim=1, keepdim=True)
            self.nlp_features[i] = nlp_features[i] / loss_norm_nlp
            loss_norm_pose = torch.norm(pose_features[i], p=2, dim=1, keepdim=True)
            self.pose_features[i] = pose_features[i] / loss_norm_pose
            loss_norm.append(torch.mean(loss_norm_pose) + torch.mean(loss_norm_nlp))
        loss_norm = torch.mean(torch.Tensor(loss_norm))

        # label loss
        nlp_length = nlp_label.size(0)
        loss_label = torch.FloatTensor(nlp_length).fill_(0).to(self.device)
        scores = torch.FloatTensor(nlp_length).fill_(0).to(self.device)
        for i in range(nlp_length):
            cur_pose = nlpid2poseid[i]
            for j in range(len(categories[0])):
                scores[i] += torch.dot(self.nlp_features[j][i], self.pose_features[j][cur_pose])*categories[i,j]
            loss_label[i] = self.get_bce_loss(scores[i], nlp_label[i])
        loss_label = torch.mean(loss_label)

        # triple loss
        loss_triple = torch.FloatTensor(len(poseid2nlpid)).fill_(0).to(self.device)
        product_matrix = self.get_product_matirx(pose_features, categories)
        num_pose = pose_features[0].size(0)
        not_find = 0
        for poseid, nlp_idx in poseid2nlpid.items():
            max_product = torch.FloatTensor(1).fill_(-1).to(self.device)
            min_product = torch.FloatTensor(1).fill_(1).to(self.device)

            for nlp_id in nlp_idx:
                if nlp_label[nlp_id] == 0:
                    if scores[nlp_id] > max_product.item():
                        max_product = scores[nlp_id]
                elif nlp_label[nlp_id] == 1:
                    if scores[nlp_id] < min_product.item():
                        min_product = scores[nlp_id]
            # pose-aware
            rand_index = torch.LongTensor([torch.rand(1)*num_pose/2]).to(self.device)
            furthest_pose = torch.argsort(product_matrix[poseid])[rand_index]
            # rand
            # furthest_pose = torch.LongTensor([torch.rand(1)*num_pose]).to(self.device)
            for nlp_id in poseid2nlpid[furthest_pose.cpu().item()]:
                if nlp_label[nlp_id] == 1:
                    cur_score = torch.zeros(1).to(self.device)
                    for j in range(len(categories[0])):
                        cur_score += torch.dot(self.nlp_features[j][nlp_id], self.pose_features[j][poseid])*categories[nlp_id,j]
                    if cur_score > max_product.item():
                        max_product = cur_score
            if max_product == -1 or min_product == 1:
                not_find += 1
                continue
            else:
                loss_triple[poseid] = max_product - min_product + 2
        if not_find > 0:
            print("not find:", not_find)
        if not_find == nlp_length:
            loss_triple = torch.FloatTensor(1).fill_(0).to(self.device)
        else:
            loss_triple = torch.sum(loss_triple)/(nlp_length-not_find)
 
        return loss_label, loss_norm, loss_triple


