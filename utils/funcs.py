############################################ IMPORT ##########################################################
import sys, os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################ EMBEDDING LOOKUP ################################################
def embedding_lookup(word_embedding, x):
    '''
    input(s) shape: [num_words, embedding_dim], [batch_size, doc_len, sen_len]
    output shape: [batch_size, doc_len, sen_len, embedding_dim]
    '''
    # x = F.embedding(torch.from_numpy(x).type(torch.LongTensor), torch.from_numpy(word_embedding))
    x = F.embedding(x.type(torch.LongTensor), word_embedding).to(device)
    return x

############################################ ATTENTION #######################################################
class Attention(nn.Module):
    def __init__(self, n_hidden, sen_len):
        super(Attention, self).__init__()
        self.n_hidden = n_hidden
        self.sen_len = sen_len
        self.linear1 = nn.Linear(n_hidden*2, n_hidden*2)
        self.linear2 = nn.Linear(n_hidden*2, 1)

    def forward(self, x):
        '''
        input shape: [batch_size * doc_len, sen_len, 2 * n_hidden]
        output shape: [batch_size * doc_len, 2 * n_hidden]
        '''
        x_tmp = x.reshape(-1, self.n_hidden*2)
        # x_tmp is of shape (batch_size * doc_len * sen_len, 2 * n_hidden)
        u = torch.tanh(self.linear1(x_tmp))
        # u is of shape (batch_size * doc_len * sen_len, 2 * n_hidden)
        alpha = self.linear2(u)
        # alpha is of shape (batch_size * doc_len * sen_len, 1)
        alpha = F.softmax(alpha.reshape(-1, 1, self.sen_len), dim = -1)
        # alpha is of shape (batch_size * doc_len, 1, sen_len)
        x = torch.matmul(alpha, x).reshape(-1, self.n_hidden*2)
        # x is of shape (batch_size * doc_len, 2 * n_hidden)
        return x

############################################## GET MASK ######################################################
def getmask(y, doc_len):
    '''
    input(s) shape: [max_doc_len * max_doc_len, 2], doc_len
    output shape: [doc_len * doc_len, 2]
    '''
    i = 0; j = 0
    max_doc_len = int(np.sqrt(y.shape[0]))
    y_mask = torch.zeros(doc_len * doc_len, 2)
    while j < doc_len**2:
        y_mask[j : j+doc_len] = y[i : i+doc_len]
        j += doc_len; i += max_doc_len
    return y_mask

############################################ LOSS FUNCTION ###################################################
class ce_loss_aux(torch.nn.Module):
    def __init__(self):
        super(ce_loss_aux, self).__init__()

    def forward(self, y_true, y_pred, doc_len, diminish_factor=1.0):
        '''
        input(s) shape: [batch_size, doc_len, 2]
        output shape: []
        '''
        y_true = y_true.to('cpu'); y_pred = y_pred.to('cpu')
        loss = torch.autograd.Variable(torch.zeros([], dtype=torch.double))
        for i in range(len(doc_len)):
            y_true_masked = y_true[i, :doc_len[i]]; y_pred_masked = y_pred[i, :doc_len[i]].double()
            y_pred_masked_ones = torch.log(y_pred_masked[:, 1][y_true_masked[:, 1]==1.])
            y_pred_masked_zeros = torch.log(y_pred_masked[:, 0][y_true_masked[:, 0]==1.])
            pos_loss = -torch.sum(y_pred_masked_ones)
            neg_loss = -torch.sum(y_pred_masked_zeros)*diminish_factor
            loss += pos_loss + neg_loss
            # loss -= torch.sum(y_true_masked * torch.log(y_pred_masked))
        loss /= torch.sum(doc_len)
        loss.requires_grad_(True)
        return loss

class ce_loss_pair(torch.nn.Module):
    def __init__(self, diminish_factor):
        super(ce_loss_pair, self).__init__()
        self.diminish_factor = diminish_factor

    def forward(self, y_true, y_pred, doc_len):
        '''
        input(s) shape: [batch_size, doc_len, 2]
        output shape: []
        '''
        y_true = y_true.to('cpu'); y_pred = y_pred.to('cpu')
        loss = torch.autograd.Variable(torch.zeros([], dtype=torch.double))
        for i in range(len(doc_len)):
            y_true_masked = getmask(y_true[i].clone(), doc_len[i])
            y_pred_masked = getmask(y_pred[i].clone(), doc_len[i]).double()
            y_pred_masked_ones = torch.log(y_pred_masked[:, 1][y_true_masked[:, 1]==1])
            pos_loss = -torch.sum(y_pred_masked_ones)
            y_pred_masked_zeros = torch.log(y_pred_masked[:, 0][y_true_masked[:, 0]==1])
            neg_loss = -torch.sum(y_pred_masked_zeros)
            ############################## Give less weight to -ve examples ##############################
            neg_loss *= self.diminish_factor
            loss += pos_loss + neg_loss
            # y_pred_masked needs to be of dtype double
        loss /= torch.sum(doc_len)
        loss.requires_grad_(True)
        return loss

########################################## CREATE PAIRS ######################################################
def create_pairs(x1, x2):
    '''
    input(s) shape: [batch_size, doc_len, 2 * n_hidden]
    output shape: [batch_size, doc_len * doc_len, 4 * n_hidden]
    '''  
    iters = 0
    for i in range(x1.shape[1]):
        for j in range(x2.shape[1]):
            x3_tmp = torch.cat([x1[:, i, :], x2[:, j, :]], dim=1).unsqueeze(1)
            if iters :
                x3 = torch.cat([x3, x3_tmp], dim=1)
            else :
                x3 = x3_tmp
            iters += 1
    return x3

############################################ DATA GEN ########################################################
def batch_index(length, batch_size, test=False):
    index = list(range(length))
    if test == False:
        np.random.shuffle(index)
    for i in range(int((length + batch_size -1)/batch_size)):
        ret = index[i * batch_size : (i + 1) * batch_size]
        if test == False and len(ret) < batch_size : break
        yield ret

def get_batch_data_pair(x, sen_len, doc_len, y_position, y_cause, y_pair, distance, batch_size, test=False):
    for index in batch_index(len(y_cause), batch_size, test):
        feed_list = [x[index], sen_len[index], doc_len[index], y_position[index], y_cause[index], \
        y_pair[index], distance[index]]
        yield feed_list, len(index)

###################################### ACCURACY, PRECISION, RECALL, F1 #######################################
def metrics(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    true_pos = np.sum((y_true==1.) & (y_pred==1.))
    true_neg = np.sum((y_true==0.) & (y_pred==0.))
    false_pos = np.sum((y_true==0.) & (y_pred==1.))
    false_neg = np.sum((y_true==1.) & (y_pred==0.))
    epsilon = 1e-9
    acc = (true_pos + true_neg)/(false_pos + false_neg + true_pos + true_neg + epsilon)
    p = true_pos/(false_pos + true_pos + epsilon)
    r = true_pos/(false_neg + true_pos + epsilon)
    f1 = 2*p*r/(p + r + epsilon)
    return acc, p, r, f1

def acc_prf_aux(pred_y, true_y, doc_len, average='weighted'):
    _, true_indices = torch.max(true_y, 2)
    _, pred_indices = torch.max(pred_y, 2)
    true_indices_masked = []; pred_indices_masked = []
    for i in range(len(doc_len)):
        true_indices_masked.extend(true_indices[i, :doc_len[i]])
        pred_indices_masked.extend(pred_indices[i, :doc_len[i]])
    # acc = precision_score(true_indices_masked, pred_indices_masked, average='micro')
    # p = precision_score(true_indices_masked, pred_indices_masked, average=average)
    # r = recall_score(true_indices_masked, pred_indices_masked, average=average)
    # f1 = f1_score(true_indices_masked, pred_indices_masked, average=average)
    acc, p, r, f1 = metrics(true_indices_masked, pred_indices_masked)
    return acc, p, r, f1
    
def acc_prf_pair(pred_y, true_y, doc_len):
    true_indices_masked_list = []; pred_indices_masked_list = []
    for i in range(len(doc_len)):
        true_y_masked = getmask(true_y[i].clone(), doc_len[i])
        pred_y_masked = getmask(pred_y[i].clone(), doc_len[i])
        _, true_indices_masked = torch.max(true_y_masked, 1)
        _, pred_indices_masked = torch.max(pred_y_masked, 1)
        # if i==len(doc_len)/2: 
        #     print(true_indices_masked); print(pred_indices_masked)
        true_indices_masked_list.extend(true_indices_masked)
        pred_indices_masked_list.extend(pred_indices_masked)
    acc, p, r, f1 = metrics(true_indices_masked_list, pred_indices_masked_list)
    return acc, p, r, f1
