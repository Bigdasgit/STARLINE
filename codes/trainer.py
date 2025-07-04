import random
import numpy as np
from model import STARLINE
import torch
from torch import optim
import math
import sys
from utility.batch_test import test_torch
from utility.load_data import Data

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed:", seed)

def set_device(gpu_id:int):
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    return device

class STARLINE_trainer: 
    def __init__(self, data_generator:Data, args:dict):
        self.device = set_device(args['gpu_id'])
        self.data_generator = data_generator
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items
        self.n_batch = data_generator.n_train // args['batch_size'] + 1
        self.feat_embed_dim = args['feat_embed_dim']
        self.lr = args['lr']
        self.batch_size = args['batch_size']
        self.n_layers = args['n_layers']
        self.has_norm = args['has_norm']
        self.decay = args['regs']
        self.alpha = args['alpha']
        self.beta = args['beta']
        self.dataset = args['dataset']
        self.agg = args['agg']
        self.target_aware = args['target_aware']
        self.cl_decay = args['cl_decay']
        self.cl_temp = args['cl_temp']

        self.adj_matrix = data_generator.adj_matrix
        self.nonzero_idx = data_generator.nonzero_idx
        nonzero_idx = torch.tensor(self.nonzero_idx).cuda().long().T
        self.rating_matrix = torch.sparse_coo_tensor(nonzero_idx, torch.ones((nonzero_idx.size(1))).cuda(), (self.n_users, self.n_items)).to_dense().cuda()
        self.users_to_test = list(data_generator.test_set.keys())
        self.users_to_val = list(data_generator.val_set.keys())
        self.image_feats = np.load(args['data_path'] + '{}/image_feat.npy'.format(self.dataset))
        self.text_feats = np.load(args['data_path'] + '{}/text_feat.npy'.format(self.dataset))
        self.gumbel_temp = args['gumbel_temp']
        self.cos_v, self.cos_t = args['cos_v'], args['cos_t']
        self.model_name = args['model_name']
        print(f"saved model name: {self.model_name}")
        
        set_seed(args['seed'])
        self.model = STARLINE(self.n_users, self.n_items, self.feat_embed_dim, self.nonzero_idx, self.has_norm, self.image_feats, self.text_feats, self.n_layers,
                                         self.alpha, self.beta, self.agg, self.ssl_temp, self.gumbel_temp, self.cos_v, self.cos_t)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def train(self):
        total_loss, bpr_loss, reg_loss, cl_loss = 0, 0, 0, 0
        
        for _ in range(self.n_batch):
            self.model.train() # set model into training mode
            self.optimizer.zero_grad()  
            user_emb, item_emb, user_emb_view, item_emb_view = self.model()
            users, pos_items, neg_items = self.data_generator.sample()
        
            batch_bpr_loss, batch_reg_loss, batch_cl_loss = self.model.bpr_infonce_loss(user_emb, item_emb, user_emb_view, item_emb_view, users, pos_items, neg_items, self.target_aware)
            batch_reg_loss = batch_reg_loss * self.decay
            batch_cl_loss = batch_cl_loss * self.cl_decay
            batch_loss = batch_bpr_loss + batch_reg_loss + batch_cl_loss
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()

            total_loss = total_loss + batch_loss.cpu().item()
            bpr_loss = bpr_loss + batch_bpr_loss.cpu().item()
            reg_loss = reg_loss + batch_reg_loss.cpu().item()
            cl_loss = cl_loss + batch_cl_loss.cpu().item()

            del user_emb, item_emb
            if self.device == torch.device('cuda'):
                torch.cuda.empty_cache()
            
        if math.isnan(total_loss):
            print("ERROR: loss is nan.")
            sys.exit()
        
        total_loss = total_loss / self.n_batch
        bpr_loss = bpr_loss / self.n_batch
        reg_loss = reg_loss / self.n_batch
        cl_loss = cl_loss / self.n_batch
        train_log = {"Total loss": total_loss, "BPR loss": bpr_loss, "Reg loss": reg_loss, "CL loss": cl_loss}
        return train_log

    def test(self, is_val=True):
        self.model.eval() # set model into evaluation mode
        with torch.no_grad():
            if is_val:
                ua_embeddings, ia_embeddings, _, _ = self.model()
            else: 
                self.model = STARLINE(self.n_users, self.n_items, self.feat_embed_dim, self.nonzero_idx, self.has_norm, self.image_feats, self.text_feats, self.n_layers,
                                         self.alpha, self.beta, self.agg, self.ssl_temp, self.gumbel_temp, self.cos_v, self.cos_t)
                self.model.load_state_dict(torch.load('saved_models/' + self.model_name, map_location='cpu', weights_only=True)[self.model_name])
                self.model.cuda()
                ua_embeddings, ia_embeddings, _, _ = self.model()
        users = self.users_to_val if is_val else self.users_to_test
        result = test_torch(ua_embeddings, ia_embeddings, users, is_val, self.rating_matrix, self.beta, self.target_aware)
        return result
    
    def save_model(self):
        torch.save({self.model_name: self.model.state_dict()}, 'saved_models/' + self.model_name)
