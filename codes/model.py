from distutils.command.build import build
from locale import normalize
import os
from re import A
import numpy as np
from time import time
from tqdm import tqdm
import random
import itertools

import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import scipy.sparse as sp
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

def normalize_laplacian(edge_index, edge_weight):

    num_nodes = maybe_num_nodes(edge_index)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    deg_inv_sqrt = deg.pow_(-0.5)
    
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_weight

class Our_GCNs(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Our_GCNs, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, weight_vector, gumbel_retain, view, size=None):
        self.isview = view
        self.gumbel_retain = gumbel_retain

        if self.isview == False:
            self.weight_vector = weight_vector
        elif self.isview == True:
            self.weight_vector = weight_vector * self.gumbel_retain

        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector
    
    def update(self, aggr_out):
        return aggr_out

class Nonlinear_GCNs(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Nonlinear_GCNs, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, weight_vector, size=None):
        x = torch.matmul(x, self.weight)
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out

class LINE(nn.Module):
    def __init__(self, nonzero_idx, feat_embed_dim, gumbel_temp=0.1, cos_v=8, cos_t=8):
        super(LINE, self).__init__()
        self.nonzero_idx = nonzero_idx
        self.feat_embed_dim = feat_embed_dim
        self.gumbel_temp = gumbel_temp
        self.mlp_v = nn.Linear(self.feat_embed_dim * 2, 2)
        self.mlp_t = nn.Linear(self.feat_embed_dim * 2, 2)
        self.cos_thr_v = (cos_v - 1) / cos_v
        self.cos_thr_t = (cos_t - 1) / cos_t

    def get_candidate_pair(self, image_preference, text_preference, image_emb, text_emb):
        image_preference = image_preference.cpu().detach().numpy()
        text_preference = text_preference.cpu().detach().numpy()
        image_emb = image_emb.cpu().detach().numpy()
        text_emb = text_emb.cpu().detach().numpy()

        img_cos = np.dot(image_preference, image_emb.T) / (np.linalg.norm(image_preference, axis=1)[:, np.newaxis] * np.linalg.norm(image_emb, axis=1)[:, np.newaxis].T)
        text_cos = np.dot(text_preference, text_emb.T) / (np.linalg.norm(text_preference, axis=1)[:, np.newaxis] * np.linalg.norm(text_emb, axis=1)[:, np.newaxis].T)

        user_indices = torch.tensor(self.nonzero_idx).cuda().long().T[0, :].cpu().detach().numpy()
        item_indices = torch.tensor(self.nonzero_idx).cuda().long().T[1, :].cpu().detach().numpy()
        img_cos[user_indices, item_indices] = -1  # excluding original edges
        text_cos[user_indices, item_indices] = -1
        
        candidate_pair_idx_img = np.argwhere(img_cos > self.cos_thr_v)  # .tolist()  # list(map(tuple, np.where(img_cos > threshold_img)))
        candidate_pair_idx_txt = np.argwhere(text_cos > self.cos_thr_t)  # .tolist()
        candidate_pair_idx_img = list(candidate_pair_idx_img)  # [tuple(v) for v in high_indices_img]
        candidate_pair_idx_txt = list(candidate_pair_idx_txt)  # [tuple(v) for v in high_indices_txt]
        
        candidate_pair_value_img = img_cos[img_cos > self.cos_thr_v]
        candidate_pair_value_txt = text_cos[text_cos > self.cos_thr_t]
        return candidate_pair_idx_img, candidate_pair_value_img, candidate_pair_idx_txt, candidate_pair_value_txt, len(candidate_pair_idx_img), len(candidate_pair_idx_txt)

    def get_edge_weight(self, gumbel_retain_v, gumbel_retain_t, candidate_pair_value_img, candidate_pair_value_txt):
        # neg to 0
        candidate_pair_value_img = torch.tensor(candidate_pair_value_img).unsqueeze(1).cuda()
        candidate_pair_value_img = candidate_pair_value_img.masked_fill(candidate_pair_value_img < 0, 0)
        candidate_pair_value_txt = torch.tensor(candidate_pair_value_txt).unsqueeze(1).cuda()
        candidate_pair_value_txt = candidate_pair_value_txt.masked_fill(candidate_pair_value_txt < 0, 0)

        # gumbel output
        gumbel_retain_v_wo_w = torch.cat([torch.unsqueeze(gumbel_retain_v, dim=1), torch.unsqueeze(gumbel_retain_v, dim=1)], dim=0)
        gumbel_retain_t_wo_w = torch.cat([torch.unsqueeze(gumbel_retain_t, dim=1), torch.unsqueeze(gumbel_retain_t, dim=1)], dim=0)

        # gumbel output with weight
        gumbel_retain_v_w = gumbel_retain_v * torch.cat([torch.ones(len(self.nonzero_idx)).cuda().view(-1, 1), candidate_pair_value_img], dim=0).squeeze()
        gumbel_retain_t_w = gumbel_retain_t * torch.cat([torch.ones(len(self.nonzero_idx)).cuda().view(-1, 1), candidate_pair_value_txt], dim=0).squeeze()

        gumbel_retain_v_w = torch.cat([torch.unsqueeze(gumbel_retain_v_w, dim=1), torch.unsqueeze(gumbel_retain_v_w, dim=1)], dim=0)
        gumbel_retain_t_w = torch.cat([torch.unsqueeze(gumbel_retain_t_w, dim=1), torch.unsqueeze(gumbel_retain_t_w, dim=1)], dim=0)

        return gumbel_retain_v_w, gumbel_retain_t_w, gumbel_retain_v_wo_w, gumbel_retain_t_wo_w

    def forward(self, image_preference, text_preference, image_emb, text_emb):

        # step1: candidate filtering for edge addition
        candidate_pair_idx_img, candidate_pair_value_img, candidate_pair_idx_txt, candidate_pair_value_txt, v_pairs, t_pairs = self.get_candidate_pair(image_preference, text_preference, image_emb, text_emb)
        self.nonzero_idx_img = self.nonzero_idx + candidate_pair_idx_img
        self.nonzero_idx_txt = self.nonzero_idx + candidate_pair_idx_txt

        # step2: edge addition and dropping
        u_nonzero_idx_img = [ui[0] for ui in self.nonzero_idx_img] 
        i_nonzero_idx_img = [ui[1] for ui in self.nonzero_idx_img] 
        u_nonzero_idx_txt = [ui[0] for ui in self.nonzero_idx_txt] 
        i_nonzero_idx_txt = [ui[1] for ui in self.nonzero_idx_txt] 

        u_embeddings_img = F.embedding(torch.tensor(u_nonzero_idx_img).cuda(), image_preference)
        i_embeddings_img = F.embedding(torch.tensor(i_nonzero_idx_img).cuda(), image_emb)
        u_embeddings_txt = F.embedding(torch.tensor(u_nonzero_idx_txt).cuda(), text_preference)
        i_embeddings_txt = F.embedding(torch.tensor(i_nonzero_idx_txt).cuda(), text_emb)

        concatenated_embeddings_v = torch.cat([u_embeddings_img, i_embeddings_img], dim=1)
        concatenated_embeddings_t = torch.cat([u_embeddings_txt, i_embeddings_txt], dim=1)

        mlp_output_v = self.mlp_v(concatenated_embeddings_v)
        mlp_output_t = self.mlp_t(concatenated_embeddings_t)

        gumbel_output_v = F.gumbel_softmax(mlp_output_v, tau=self.gumbel_temp, hard=True)[:, :]
        gumbel_output_t = F.gumbel_softmax(mlp_output_t, tau=self.gumbel_temp, hard=True)[:, :]

        gumbel_retain_v = gumbel_output_v[:, 0]
        gumbel_retain_t = gumbel_output_t[:, 0]

        # step3: edge weight assignment
        gumbel_retain_v_w, gumbel_retain_t_w, gumbel_retain_v_wo_w, gumbel_retain_t_wo_w = self.get_edge_weight(gumbel_retain_v, gumbel_retain_t, candidate_pair_value_img, candidate_pair_value_txt)

        return gumbel_retain_v_w, gumbel_retain_t_w, gumbel_retain_v_wo_w, gumbel_retain_t_wo_w, self.nonzero_idx_img, self.nonzero_idx_txt, v_pairs, t_pairs

class MeGCN(nn.Module):
    def __init__(self, n_users, n_items, n_layers, has_norm, feat_embed_dim, nonzero_idx, image_feats, text_feats, alpha, agg):
        super(MeGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.has_norm = has_norm
        self.feat_embed_dim = feat_embed_dim
        self.nonzero_idx = torch.tensor(nonzero_idx).cuda().long().T
        self.alpha = alpha
        self.agg = agg

        self.image_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
        self.text_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
        nn.init.xavier_uniform_(self.image_preference.weight)                    
        nn.init.xavier_uniform_(self.text_preference.weight)

        self.image_embedding = nn.Embedding.from_pretrained(torch.tensor(image_feats, dtype=torch.float), freeze=True) # [# of items, 4096]
        self.text_embedding = nn.Embedding.from_pretrained(torch.tensor(text_feats, dtype=torch.float), freeze=True) # [# of items, 1024]

        self.image_trs = nn.Linear(image_feats.shape[1], self.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], self.feat_embed_dim)
        
        if self.agg == 'fc':
            self.transform = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
        elif self.agg == 'weighted_sum':
            self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
            self.softmax = nn.Softmax(dim=0)
            
        self.layers = nn.ModuleList([Our_GCNs(self.feat_embed_dim, self.feat_embed_dim) for _ in range(self.n_layers)])

    def forward(self, edge_index_img, edge_weight_img, edge_index_txt, edge_weight_txt, gumbel_retain_img=None, gumbel_retain_txt=None, view=False, eval=False):  # V, T  # gumbel_retain_img, gumbel_retain_txt

        # transform
        image_emb = self.image_trs(self.image_embedding.weight) # [# of items, feat_embed_dim]
        text_emb = self.text_trs(self.text_embedding.weight) # [# of items, feat_embed_dim]
        
        if self.has_norm:
            image_emb = F.normalize(image_emb)
            text_emb = F.normalize(text_emb)
        image_preference = self.image_preference.weight
        text_preference = self.text_preference.weight

        # view
        self.isview = view

        # propagate
        ego_image_emb = torch.cat([image_preference, image_emb], dim=0)
        ego_text_emb = torch.cat([text_preference, text_emb], dim=0)

        for layer in self.layers:
            side_image_emb = layer(ego_image_emb, edge_index_img, edge_weight_img, gumbel_retain_img, self.isview)  # layer(ego_image_emb, edge_index_img, edge_weight_img)  # layer(ego_image_emb, edge_index_img, edge_weight_img, gumbel_retain_img)
            side_text_emb = layer(ego_text_emb, edge_index_txt, edge_weight_txt, gumbel_retain_txt, self.isview)  # layer(ego_text_emb, edge_index_txt, edge_weight_txt)  # layer(ego_text_emb, edge_index_txt, edge_weight_txt, gumbel_retain_txt)

            ego_image_emb = side_image_emb + self.alpha * ego_image_emb
            ego_text_emb = side_text_emb + self.alpha * ego_text_emb

        final_image_preference, final_image_emb = torch.split(ego_image_emb, [self.n_users, self.n_items], dim=0)
        final_text_preference, final_text_emb = torch.split(ego_text_emb, [self.n_users, self.n_items], dim=0)
        
        if eval:
            return ego_image_emb, ego_text_emb

        if self.agg == 'concat':
            items = torch.cat([final_image_emb, final_text_emb], dim=1) # [# of items, feat_embed_dim * 2]
            user_preference = torch.cat([final_image_preference, final_text_preference], dim=1) # [# of users, feat_embed_dim * 2]
        elif self.agg == 'sum':
            items = final_image_emb + final_text_emb # [# of items, feat_embed_dim]
            user_preference = final_image_preference + final_text_preference # [# of users, feat_embed_dim]
        elif self.agg == 'weighted_sum':
            weight = self.softmax(self.modal_weight)
            items = weight[0] * final_image_emb + weight[1] * final_text_emb # [# of items, feat_embed_dim]
            user_preference = weight[0] * final_image_preference + weight[1] * final_text_preference # [# of users, feat_embed_dim]
        elif self.agg == 'fc':
            items = self.transform(torch.cat([final_image_emb, final_text_emb], dim=1)) # [# of items, feat_embed_dim]
            user_preference = self.transform(torch.cat([final_image_preference, final_text_preference], dim=1)) # [# of users, feat_embed_dim]
        
        return user_preference, items

class STARLINE(nn.Module):
    def __init__(self, n_users, n_items, feat_embed_dim, nonzero_idx, has_norm, image_feats, text_feats, n_layers, alpha, beta, agg, ssl_temp, gumbel_temp, cos_v, cos_t):
        super(STARLINE, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.feat_embed_dim = feat_embed_dim
        self.n_layers = n_layers
        self.nonzero_idx = nonzero_idx
        self.alpha = alpha
        self.beta = beta
        self.agg = agg
        self.image_feats = torch.tensor(image_feats, dtype=torch.float).cuda()
        self.text_feats = torch.tensor(text_feats, dtype=torch.float).cuda()

        self.megcn = MeGCN(self.n_users, self.n_items, self.n_layers, has_norm, self.feat_embed_dim, self.nonzero_idx, image_feats, text_feats, self.alpha, self.agg)
        
        ## LINE ##
        self.gumbel_temp = gumbel_temp  ## gumbel tau ##
        self.cos_v, self.cos_t = cos_v, cos_t
        self.line = LINE(self.nonzero_idx, self.feat_embed_dim, self.gumbel_temp, self.cos_v, self.cos_t)  ## LINE ##
        ##########
        
        nonzero_idx = torch.tensor(self.nonzero_idx).cuda().long().T
        nonzero_idx[1] = nonzero_idx[1] + self.n_users

        self.edge_index = torch.cat([nonzero_idx, torch.stack([nonzero_idx[1], nonzero_idx[0]], dim=0)], dim=1)
        self.edge_weight = torch.ones((self.edge_index.size(1))).cuda().view(-1, 1)
        self.edge_weight = normalize_laplacian(self.edge_index, self.edge_weight)

        nonzero_idx = torch.tensor(self.nonzero_idx).cuda().long().T
        self.adj = torch.sparse_coo_tensor(nonzero_idx, torch.ones((nonzero_idx.size(1))).cuda(), (self.n_users, self.n_items)).to_dense().cuda()

        ## cl ##
        self.ssl_temp = ssl_temp
        ########

    def forward(self, eval=False):

        ## LINE ##
        image_emb = self.megcn.image_trs(self.megcn.image_embedding.weight) # [# of items, feat_embed_dim]
        text_emb = self.megcn.text_trs(self.megcn.text_embedding.weight) # [# of items, feat_embed_dim]
        
        if self.megcn.has_norm:
            image_emb = F.normalize(image_emb)
            text_emb = F.normalize(text_emb)
        image_preference = self.megcn.image_preference.weight
        text_preference = self.megcn.text_preference.weight

        ## LINE ##
        gumbel_retain_v_w, gumbel_retain_t_w, gumbel_retain_v_wo_w, gumbel_retain_t_wo_w, pair_retain_img, pair_retain_txt, v_pairs, t_pairs = self.line(image_preference, text_preference, image_emb, text_emb)
        ##########

        if eval:
            ## cl ##
            img, txt = self.megcn(self.edge_index, self.edge_weight, self.edge_index, self.edge_weight, view=False, eval=True)
            ########
            return img, txt
        
        ## original ##
        user, items = self.megcn(self.edge_index, self.edge_weight, self.edge_index, self.edge_weight, view=False, eval=False)
        ##############

        ## cl ##
        pair_retain_img = torch.tensor(np.array(pair_retain_img)).cuda().long().T
        pair_retain_img[1] = pair_retain_img[1] + self.n_users
        self.retained_edge_index_img = torch.cat([pair_retain_img, torch.stack([pair_retain_img[1], pair_retain_img[0]], dim=0)], dim=1)  # torch.Size([2, 2489144])

        pair_retain_txt = torch.tensor(np.array(pair_retain_txt)).cuda().long().T
        pair_retain_txt[1] = pair_retain_txt[1] + self.n_users
        self.retained_edge_index_txt = torch.cat([pair_retain_txt, torch.stack([pair_retain_txt[1], pair_retain_txt[0]], dim=0)], dim=1)  # torch.Size([2, 2489144])
        ########

        ## cl ##
        self.edge_weight_view_img = torch.ones((self.retained_edge_index_img.size(1))).cuda().view(-1, 1) * gumbel_retain_v_w + 1e-9
        self.edge_weight_view_txt = torch.ones((self.retained_edge_index_txt.size(1))).cuda().view(-1, 1) * gumbel_retain_t_w + 1e-9
        self.edge_weight_view_img = normalize_laplacian(self.retained_edge_index_img, self.edge_weight_view_img)  # + 1e-9  # * gumbel_output_img_without_w + 1e-9
        self.edge_weight_view_txt = normalize_laplacian(self.retained_edge_index_txt, self.edge_weight_view_txt)  # + 1e-9  # * gumbel_output_txt_without_w + 1e-9
        user_view, items_view = self.megcn(self.retained_edge_index_img, self.edge_weight_view_img, self.retained_edge_index_txt, self.edge_weight_view_txt, gumbel_retain_v_wo_w, gumbel_retain_t_wo_w, view=True, eval=False)
        ########
        # return user, items
        return user, items, user_view, items_view
        
    def bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items, target_aware):

        # print('BPR loss!')

        batch_size = len(users)
        current_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]

        if target_aware:
            # target-aware
            item_item = torch.mm(item_emb, item_emb.T)
            pos_item_query = item_item[pos_items, :]  # (batch_size, n_items)
            neg_item_query = item_item[neg_items, :]  # (batch_size, n_items)
            pos_target_user_alpha = torch.softmax(torch.multiply(pos_item_query, self.adj[users, :]).masked_fill(self.adj[users, :] == 0, -1e9), dim=1)  # (batch_size, n_items)
            neg_target_user_alpha = torch.softmax(torch.multiply(neg_item_query, self.adj[users, :]).masked_fill(self.adj[users, :] == 0, -1e9), dim=1)  # (batch_size, n_items)
            pos_target_user = torch.mm(pos_target_user_alpha, item_emb)  # (batch_size, dim) 
            neg_target_user = torch.mm(neg_target_user_alpha, item_emb)  # (batch_size, dim) 

            # predictor
            pos_scores = (1 - self.beta) * torch.sum(torch.mul(current_user_emb, pos_item_emb), dim=1) + self.beta * torch.sum(torch.mul(pos_target_user, pos_item_emb), dim=1)
            neg_scores = (1 - self.beta) * torch.sum(torch.mul(current_user_emb, neg_item_emb), dim=1) + self.beta * torch.sum(torch.mul(neg_target_user, neg_item_emb), dim=1)
        else:
            pos_scores = torch.sum(torch.mul(current_user_emb, pos_item_emb), dim=1)
            neg_scores = torch.sum(torch.mul(current_user_emb, neg_item_emb), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        regularizer = 1./2 * (pos_item_emb**2).sum() + 1./2 * (neg_item_emb**2).sum() + 1./2 * (current_user_emb**2).sum()
        reg_loss = regularizer / pos_item_emb.size(0)

        return mf_loss, reg_loss
    
    ## add ##
    def inner_product(self, a, b):
        return torch.sum(a * b, dim=-1)
    
    # def infonce_loss(self, item_embeddings1, items):  # items -> pos_items
    def infonce_loss(self, user_embeddings1, item_embeddings1, user_embeddings_refined, item_embeddings_refined, users, items, neg_items):

        ## original - view2 ##
        user_embeddings1 = F.normalize(user_embeddings1, dim=1) 
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings_refined = F.normalize(user_embeddings_refined, dim=1)
        item_embeddings_refined = F.normalize(item_embeddings_refined, dim=1)

        users = torch.tensor(users).cuda()
        items = torch.tensor(items).cuda()
        user_embs_original = F.embedding(users, user_embeddings1)  # torch.Size([1024, 128])
        item_embs_original = F.embedding(items, item_embeddings1)  # torch.Size([1024, 128])
        user_embs_refined = F.embedding(users, user_embeddings_refined)
        item_embs_refined = F.embedding(items, item_embeddings_refined)

        pos_ratings_user = self.inner_product(user_embs_original, user_embs_refined)
        pos_ratings_item = self.inner_product(item_embs_original, item_embs_refined)
        tot_ratings_user = torch.matmul(user_embs_original, 
                                        torch.transpose(user_embeddings_refined, 0, 1))    
        tot_ratings_item = torch.matmul(item_embs_original, 
                                        torch.transpose(item_embeddings_refined, 0, 1)) 

        ssl_logits_user_refined = tot_ratings_user - pos_ratings_user[:, None]                
        ssl_logits_item_refined = tot_ratings_item - pos_ratings_item[:, None] 

        clogits_user_refined = torch.logsumexp(ssl_logits_user_refined / self.ssl_temp, dim=1)  
        clogits_item_refined = torch.logsumexp(ssl_logits_item_refined / self.ssl_temp, dim=1)
        ######################

        ## cl ##
        # infonce_loss = torch.mean(clogits_item + clogits_user_refined + clogits_item_refined)  # clogits_item + clogits_user_refined + clogits_item_refined
        infonce_loss = (1/2) * (torch.mean(clogits_user_refined) + torch.mean(clogits_item_refined))
        ########
    
        return infonce_loss
    
    # def bpr_infonce_loss(self, user_emb, item_emb, users, pos_items, neg_items, target_aware):
    def calculate_loss(self, user_emb, item_emb, user_emb_view, item_emb_view, users, pos_items, neg_items, target_aware):

        mf_loss, reg_loss = self.bpr_loss(user_emb, item_emb, users, pos_items, neg_items, target_aware)
        # infonce_loss = self.infonce_loss(user_emb, item_emb, user_emb_view, item_emb_view, users, pos_items, neg_items)
        infonce_loss = self.infonce_loss(user_emb, item_emb, user_emb_view, item_emb_view, users, pos_items, neg_items)

        return mf_loss, reg_loss, infonce_loss
    #########
