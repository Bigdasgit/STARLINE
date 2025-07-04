import utility.metrics as metrics
from utility.parser import args
from utility.load_data import data_generator
import multiprocessing
import heapq
import torch
import numpy as np
from time import time
from tqdm import tqdm

cores = multiprocessing.cpu_count() - 2

Ks = eval(args['Ks'])

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
if args['target_aware']:
    BATCH_SIZE = 16
else:
    BATCH_SIZE = args['batch_size']

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    is_val = x[-1]
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    if is_val:
        user_pos_test = data_generator.val_set[u]
    else:
        user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args['test_flag'] == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val, adj, beta, target_aware):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0

    # user_user = torch.mm(ua_embeddings, ua_embeddings.T)
    item_item = torch.mm(ia_embeddings, ia_embeddings.T)

    for u_batch_id in tqdm(range(n_user_batchs)):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]
        if target_aware:
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)
                u_g_embeddings = ua_embeddings[user_batch] # (batch_size, dim)
                i_g_embeddings = ia_embeddings[item_batch] # (batch_size, dim)

                # # comparison target (mean pooling)
                # neighbor_user = torch.mm(adj[user_batch, :], ia_embeddings) # (batch_size, dim)
                # neighbor_user = torch.div(neighbor_user, adj[user_batch, :].sum(dim=1).unsqueeze(1))

                # user side target-aware (best target-aware)
                item_query = item_item[item_batch, :] # (item_batch_size, n_items)
                item_target_user_alpha = torch.softmax(torch.multiply(item_query.unsqueeze(1), adj[user_batch, :].unsqueeze(0)).masked_fill(adj[user_batch, :].repeat(len(item_batch), 1, 1) == 0, -1e9), dim=2) # (item_batch_size, user_batch_size, n_items)
                item_target_user = torch.matmul(item_target_user_alpha, ia_embeddings) # (item_batch_size, user_batch_size, dim)
                
                # # # item side target-aware
                # # user_query = user_user[user_batch, :] # (user_batch_size, n_users)
                # # user_target_item_alpha = torch.softmax(torch.multiply(user_query.unsqueeze(1), adj[:, item_batch].T.unsqueeze(0)).masked_fill(adj[:, item_batch].T.repeat(len(user_batch), 1, 1) == 0, -1e9), dim=2) # (user_batch_size, item_batch_size, n_users)
                # # user_target_item = torch.matmul(user_target_item_alpha, ua_embeddings) # (user_batch_size, item_batch_size, dim)

                # user side target-aware (best target-aware)
                i_rate_batch = (1 - beta) * torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1)) + beta * torch.sum(torch.mul(item_target_user.permute(1, 0, 2).contiguous(), i_g_embeddings), dim=2) # torch.matmul(neighbor_user, torch.transpose(i_g_embeddings, 0, 1))

                rate_batch[:, i_start: i_end] = i_rate_batch.detach().cpu().numpy()
                i_count += i_rate_batch.shape[1]

                del item_query, item_target_user_alpha, item_target_user, i_g_embeddings, u_g_embeddings #, neighbor_user
                torch.cuda.empty_cache()

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]

            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))
            rate_batch = rate_batch.detach().cpu().numpy()

        user_batch_rating_uid = zip(rate_batch, user_batch, [is_val] * len(user_batch))

        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users

    assert count == n_test_users
    pool.close()
    return result
