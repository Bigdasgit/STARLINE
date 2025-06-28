import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, diags
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def normalize_adj_mat(adj_matrix:coo_matrix):
    rowsum = np.array(adj_matrix.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_matrix = diags(d_inv)
    
    sparse_mx = d_matrix.dot(adj_matrix)
    sparse_mx = sparse_mx.dot(d_matrix)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
    
class LINE(nn.Module):
    def __init__(self,
                 adj_matrix:coo_matrix, 
                 feat_embed_dim:int,
                 gumbel_temp:float,
                 cos_v:float,
                 cos_t:float
        ):
        super(LINE, self).__init__()
        self.adj_matrix = adj_matrix
        self.feat_embed_dim = feat_embed_dim
        self.gumbel_temp = gumbel_temp
        
        self.cos_thr_v = self.get_threshold(cos_v)
        self.cos_thr_t = self.get_threshold(cos_t)
        print(f"cos threshold for visual feature: {self.cos_thr_v}")
        print(f"cos threshold for text feature: {self.cos_thr_t}")
        
        self.mlp_v = nn.Linear(self.feat_embed_dim * 2, 2) 
        self.mlp_t = nn.Linear(self.feat_embed_dim * 2, 2)
    
    def get_threshold(self, cos_interval:int):
        cos_interval = cos_interval // 2
        return (cos_interval - 1) / cos_interval
        
    def get_candidate_pair(self, image_preference:torch.Tensor, text_preference:torch.Tensor, image_emb:torch.Tensor, text_emb:torch.Tensor):
        image_preference = image_preference.cpu()
        text_preference = text_preference.cpu()
        image_emb = image_emb.cpu()
        text_emb = text_emb.cpu()
        
        # cos similarity of all (u, i) pairs in visual embedding
        image_preference_norm = F.normalize(image_preference, p=2, dim=1)
        image_emb_norm = F.normalize(image_emb, p=2, dim=1)
        img_cos = image_preference_norm @ image_emb_norm.T

        # cos similarity of all (u, i) pairs in text embedding        
        text_preference_norm = F.normalize(text_preference, p=2, dim=1)
        text_emb_norm = F.normalize(text_emb, p=2, dim=1)
        text_cos = text_preference_norm @ text_emb_norm.T
        
        candidate_pair_idx_img = torch.argwhere(img_cos > self.cos_thr_v).detach().numpy()
        candidate_pair_idx_txt = torch.argwhere(text_cos > self.cos_thr_t).detach().numpy()
        candidate_pair_value_img = img_cos[img_cos > self.cos_thr_v].detach().numpy()
        candidate_pair_value_txt = text_cos[text_cos > self.cos_thr_t].detach().numpy()
        
        return candidate_pair_idx_img, candidate_pair_idx_txt, candidate_pair_value_img, candidate_pair_value_txt
        
    def forward(self, image_preference:torch.Tensor, text_preference:torch.Tensor, image_emb:torch.Tensor, text_emb:torch.Tensor):
        n_users = image_preference.size(0)
        candidate_pair_idx_img, candidate_pair_idx_txt, candidate_pair_value_img, candidate_pair_value_txt = self.get_candidate_pair(image_preference, text_preference, image_emb, text_emb)
        
        old_row = self.adj_matrix.row
        old_col = self.adj_matrix.col
        old_data = self.adj_matrix.data
        shape = self.adj_matrix.shape
        
        users_to_add_img = candidate_pair_idx_img[:, 0]
        items_to_add_img = candidate_pair_idx_img[:, 1] + n_users
        
        users_to_add_txt = candidate_pair_idx_txt[:, 0]
        items_to_add_txt = candidate_pair_idx_txt[:, 1] + n_users
        
        new_row_img = np.concatenate([old_row, users_to_add_img, items_to_add_img])
        new_col_img = np.concatenate([old_col, items_to_add_img, users_to_add_img])
        new_data_img = np.concatenate([old_data, candidate_pair_value_img, candidate_pair_value_img])
        
        new_row_txt = np.concatenate([old_row, users_to_add_txt, items_to_add_txt])
        new_col_txt = np.concatenate([old_col, items_to_add_txt, users_to_add_txt])
        new_data_txt = np.concatenate([old_data, candidate_pair_value_txt, candidate_pair_value_txt])
        
        all_img_emb = torch.cat([image_preference, image_emb], dim=0)
        all_txt_emb = torch.cat([text_preference, text_emb], dim=0) 

        # refine visual adj matrix
        mask_v = new_row_img < new_col_img
        source_v = new_row_img[mask_v]
        dest_v = new_col_img[mask_v]
        value_v = new_data_img[mask_v]

        source_emb_v = all_img_emb[source_v]
        dest_emb_v = all_img_emb[dest_v]
        concatednated_embeddings_v = torch.cat([source_emb_v, dest_emb_v], dim=1)
        mlp_out_v = self.mlp_v(concatednated_embeddings_v)
        gumbel_out_v = F.gumbel_softmax(mlp_out_v, tau=self.gumbel_temp, hard=True)
        gumbel_retain_v = gumbel_out_v[:, 0].detach().cpu().numpy()

        refined_source_v = source_v[gumbel_retain_v == 1]
        refined_dest_v = dest_v[gumbel_retain_v == 1]
        refined_value_v = value_v[gumbel_retain_v == 1]

        refined_row_v = np.concatenate([refined_source_v, refined_dest_v])
        refined_col_v = np.concatenate([refined_dest_v, refined_source_v])
        refined_data_v = np.concatenate([refined_value_v, refined_value_v])
        refined_adj_matrix_img = coo_matrix((refined_data_v, (refined_row_v, refined_col_v)), shape=shape)

        # refine text adj matrix
        mask_t = new_row_txt < new_col_txt
        source_t = new_row_txt[mask_t]
        dest_t = new_col_txt[mask_t]
        value_t = new_data_txt[mask_t]

        source_emb_t = all_txt_emb[source_t]
        dest_emb_t = all_txt_emb[dest_t]
        concatednated_embeddings_t = torch.cat([source_emb_t, dest_emb_t], dim=1)
        mlp_out_t = self.mlp_t(concatednated_embeddings_t)
        gumbel_out_t = F.gumbel_softmax(mlp_out_t, tau=self.gumbel_temp, hard=True)
        gumbel_retain_t = gumbel_out_t[:, 0].detach().cpu().numpy()
        
        refined_value_t = value_t[gumbel_retain_t == 1]
        refined_source_t = source_t[gumbel_retain_t == 1]
        refined_dest_t = dest_t[gumbel_retain_t == 1]

        refined_row_t = np.concatenate([refined_source_t, refined_dest_t])
        refined_col_t = np.concatenate([refined_dest_t, refined_source_t])
        refined_data_t = np.concatenate([refined_value_t, refined_value_t])
        refined_adj_matrix_txt = coo_matrix((refined_data_t, (refined_row_t, refined_col_t)), shape=shape)

        refined_norm_adj_matrix_img = normalize_adj_mat(refined_adj_matrix_img).coalesce().cuda()
        refined_norm_adj_matrix_txt = normalize_adj_mat(refined_adj_matrix_txt).coalesce().cuda()
        
        return refined_norm_adj_matrix_img, refined_norm_adj_matrix_txt

class MeGCN(nn.Module):
    def __init__(self,
                n_users:int,
                n_items:int,
                n_layers:int,
                has_norm:bool,
                feat_embed_dim:int,
                adj_matrix:coo_matrix,
                rating_matrix:torch.FloatTensor,
                image_feats:np.ndarray,
                text_feats:np.ndarray,
                alpha:float,
                beta:float
        ):
        super(MeGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.has_norm = has_norm
        self.feat_embed_dim = feat_embed_dim
        self.adj_matrix = adj_matrix
        self.norm_adj_matrix = normalize_adj_mat(self.adj_matrix).coalesce().cuda()
        self.rating_matrix = rating_matrix
        self.image_feats = torch.tensor(image_feats, dtype=torch.float).cuda()
        self.text_feats = torch.tensor(text_feats, dtype=torch.float).cuda()
        self.alpha = alpha
        self.beta = beta

        self.image_preference = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.feat_embed_dim)
        self.text_preference = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.feat_embed_dim)
        nn.init.xavier_uniform_(self.image_preference.weight)
        nn.init.xavier_uniform_(self.text_preference.weight)
        self.image_embedding = nn.Embedding.from_pretrained(torch.tensor(image_feats, dtype=torch.float), freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(torch.tensor(text_feats, dtype=torch.float), freeze=True)
        self.image_trs = nn.Linear(image_feats.shape[1], self.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], self.feat_embed_dim)
    
    def forward(self, view:bool=False, refined_norm_adj_matrix_img:torch.FloatTensor=None, refined_norm_adj_matrix_txt:torch.FloatTensor=None):
        # MeGCN
        image_emb = self.image_trs(self.image_embedding.weight)
        text_emb = self.text_trs(self.text_embedding.weight)
        if self.has_norm:
            image_emb = F.normalize(image_emb)
            text_emb = F.normalize(text_emb)
        image_preference = self.image_preference.weight
        text_preference = self.text_preference.weight
        # propagate
        ego_image_emb = torch.cat([image_preference, image_emb], dim=0)
        ego_text_emb = torch.cat([text_preference, text_emb], dim=0)

        for layer in range(self.n_layers):
            if view:
                side_image_emb = torch.sparse.mm(refined_norm_adj_matrix_img, ego_image_emb)
                side_text_emb = torch.sparse.mm(refined_norm_adj_matrix_txt, ego_text_emb)    
            else:
                side_image_emb = torch.sparse.mm(self.norm_adj_matrix, ego_image_emb)
                side_text_emb = torch.sparse.mm(self.norm_adj_matrix, ego_text_emb)

            ego_image_emb = side_image_emb + self.alpha * ego_image_emb
            ego_text_emb = side_text_emb + self.alpha * ego_text_emb

        final_image_preference, final_image_emb = torch.split(ego_image_emb, [self.n_users, self.n_items], dim=0)
        final_text_preference, final_text_emb = torch.split(ego_text_emb, [self.n_users, self.n_items], dim=0)

        items = torch.cat([final_image_emb, final_text_emb], dim=1)
        user_preference = torch.cat([final_image_preference, final_text_preference], dim=1)

        return user_preference, items

class STARLINE(nn.Module):        
    def __init__(self, 
                n_users:int, 
                n_items:int, 
                n_layers:int,
                has_norm:bool, 
                feat_embed_dim:int, 
                adj_matrix:coo_matrix,
                rating_matrix:torch.FloatTensor,
                image_feats:np.ndarray, 
                text_feats:np.ndarray, 
                alpha:float, 
                beta:float, 
                ssl_temp:float, 
                gumbel_temp:float, 
                cos_v:float, 
                cos_t:float
        ):
        super(STARLINE, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.has_norm = has_norm
        self.feat_embed_dim = feat_embed_dim
        self.n_layers = n_layers
        self.alpha = alpha
        self.beta = beta
        self.image_feats = image_feats
        self.text_feats = text_feats  
        self.adj_matrix = adj_matrix
        self.rating_matrix = rating_matrix
    
        self.megcn = MeGCN(self.n_users, self.n_items, self.n_layers, self.has_norm, self.feat_embed_dim, self.adj_matrix, self.rating_matrix, self.image_feats, self.text_feats, self.alpha, self.beta)
            
        ## LINE ##
        self.gumbel_temp = gumbel_temp  ## gumbel tau ##
        self.cos_v, self.cos_t = cos_v, cos_t
        self.line = LINE(self.adj_matrix, self.feat_embed_dim, self.gumbel_temp, self.cos_v, self.cos_t)
        ##########

        ## cl ##
        self.ssl_temp = ssl_temp
        ########

    def forward(self):
        
        ## original ##
        user, items = self.megcn()
        ##############
        
        ## LINE ##
        image_emb = self.megcn.image_trs(self.megcn.image_embedding.weight) # [# of items, feat_embed_dim]
        text_emb = self.megcn.text_trs(self.megcn.text_embedding.weight) # [# of items, feat_embed_dim]
        if self.megcn.has_norm:
            image_emb = F.normalize(image_emb)
            text_emb = F.normalize(text_emb)
        image_preference = self.megcn.image_preference.weight
        text_preference = self.megcn.text_preference.weight
        
        refined_norm_adj_matrix_img, refined_norm_adj_matrix_txt = self.line(image_preference, text_preference, image_emb, text_emb)
        ##########
        
        ## view ##
        user_view, items_view = self.megcn(view=True, refined_norm_adj_matrix_img=refined_norm_adj_matrix_img, refined_norm_adj_matrix_txt=refined_norm_adj_matrix_txt)
        ##########        

        return user, items, user_view, items_view
        
    def bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items, target_aware):
        current_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]

        if target_aware:
            # target-aware
            item_item = torch.mm(item_emb, item_emb.T)
            pos_item_query = item_item[pos_items, :]  # (batch_size, n_items)
            neg_item_query = item_item[neg_items, :]  # (batch_size, n_items)
            pos_target_user_alpha = torch.softmax(torch.multiply(pos_item_query, self.rating_matrix[users, :]).masked_fill(self.rating_matrix[users, :] == 0, -1e9), dim=1)  # (batch_size, n_items)
            neg_target_user_alpha = torch.softmax(torch.multiply(neg_item_query, self.rating_matrix[users, :]).masked_fill(self.rating_matrix[users, :] == 0, -1e9), dim=1)  # (batch_size, n_items)
            pos_target_user = torch.mm(pos_target_user_alpha, item_emb)  # (batch_size, dim) 
            neg_target_user = torch.mm(neg_target_user_alpha, item_emb)  # (batch_size, dim) 

            # predictor
            pos_scores = (1 - self.beta) * torch.sum(torch.mul(current_user_emb, pos_item_emb), dim=1) + self.beta * torch.sum(torch.mul(pos_target_user, pos_item_emb), dim=1)
            neg_scores = (1 - self.beta) * torch.sum(torch.mul(current_user_emb, neg_item_emb), dim=1) + self.beta * torch.sum(torch.mul(neg_target_user, neg_item_emb), dim=1)
        else:
            pos_scores = torch.sum(torch.mul(current_user_emb, pos_item_emb), dim=1)
            neg_scores = torch.sum(torch.mul(current_user_emb, neg_item_emb), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        bpr_loss = -torch.mean(maxi)

        regularizer = 1./2 * (
            torch.pow(pos_item_emb, 2).sum() +
            torch.pow(neg_item_emb, 2).sum() +
            torch.pow(current_user_emb, 2).sum()
        )
        reg_loss = regularizer / pos_item_emb.size(0)

        return bpr_loss, reg_loss
    
    ## add ##
    def inner_product(self, a, b):
        return torch.sum(a * b, dim=-1)
    
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
    
    def bpr_infonce_loss(self, user_emb, item_emb, user_emb_view, item_emb_view, users, pos_items, neg_items, target_aware):

        bpr_loss, reg_loss = self.bpr_loss(user_emb, item_emb, users, pos_items, neg_items, target_aware)
        # infonce_loss = self.infonce_loss(user_emb, item_emb, user_emb_view, item_emb_view, users, pos_items, neg_items)
        infonce_loss = self.infonce_loss(user_emb, item_emb, user_emb_view, item_emb_view, users, pos_items, neg_items)

        return bpr_loss, reg_loss, infonce_loss
    #########
