############################################ IMPORT ##########################################################
import sys, os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from utils.funcs import *
from utils.prepare_data import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################ FLAGS ############################################################
train_file_path = './data_combine_eng/clause_keywords.csv'          # clause keyword file
w2v_file = './data_combine_eng/w2v_200.txt'                         # embedding file
embedding_dim = 200                                                 # dimension of word embedding
embedding_dim_pos = 50                                              # dimension of position embedding
max_sen_len = 30                                                    # max number of tokens per sentence
max_doc_len = 41                                                    # max number of tokens per document
n_hidden = 100                                                      # number of hidden unit
n_class = 2                                                         # number of distinct class
training_epochs = 15                                                # number of train epochs
batch_size = 32                                                     # number of example per batch
learning_rate = 0.0050                                              # learning rate
keep_prob1 = 0.8                                                    # word embedding training dropout keep prob
keep_prob2 = 1.0                                                    # softmax layer dropout keep prob
keep_prob3 = 1.0                                                    # softmax layer dropout keep prob
l2_reg = 0.00010                                                    # l2 regularization
cause = 1.0                                                         # lambda1
pos = 1.0                                                           # lambda2
pair = 2.5                                                          # lambda3
diminish_factor = 0.400                                             # give less weight to -ve examples

############################################ MODEL ############################################################
class E2E_PextE(nn.Module):
    def __init__(self, embedding_dim, embedding_dim_pos, sen_len, doc_len, keep_prob1, keep_prob2, \
                 keep_prob3, n_hidden, n_class):
        super(E2E_PextE, self).__init__()
        self.embedding_dim = embedding_dim; self.embedding_dim_pos = embedding_dim_pos 
        self.sen_len = sen_len; self.doc_len = doc_len
        self.keep_prob1 = keep_prob1; self.keep_prob2 = keep_prob2
        self.n_hidden = n_hidden; self.n_class = n_class

        self.dropout1 = nn.Dropout(p = 1 - keep_prob1)
        self.dropout2 = nn.Dropout(p = 1 - keep_prob2)
        self.dropout3 = nn.Dropout(p = 1 - keep_prob3)
        self.relu = nn.ReLU()
        self.pos_linear = nn.Linear(2*n_hidden, n_class)
        self.cause_linear = nn.Linear(2*n_hidden, n_class)
        self.pair_linear1 = nn.Linear(4*n_hidden + embedding_dim_pos, n_hidden//2)
        self.pair_linear2 = nn.Linear(n_hidden//2, n_class)
        self.word_bilstm = nn.LSTM(embedding_dim, n_hidden, batch_first = True, bidirectional = True)
        self.cause_bilstm = nn.LSTM(2*n_hidden + n_class, n_hidden, batch_first = True, bidirectional = True)
        self.pos_bilstm = nn.LSTM(2*n_hidden, n_hidden, batch_first = True, bidirectional = True)
        self.attention = Attention(n_hidden, sen_len)

    def get_clause_embedding(self, x):
        '''
        input shape: [batch_size, doc_len, sen_len, embedding_dim]
        output shape: [batch_size, doc_len, 2 * n_hidden]
        '''
        x = x.reshape(-1, self.sen_len, self.embedding_dim)
        x = self.dropout1(x)
        # x is of shape (batch_size * max_doc_len, max_sen_len, embedding_dim)
        x, hidden_states = self.word_bilstm(x.float())
        # x is of shape (batch_size * max_doc_len, max_sen_len, 2 * n_hidden)
        s = self.attention(x).reshape(-1, self.doc_len, 2 * self.n_hidden)
        # s is of shape (batch_size, max_doc_len, 2 * n_hidden)
        return s

    def get_emotion_prediction(self, x):
        '''
        input shape: [batch_size, doc_len, 2 * n_hidden]
        output(s) shape: [batch_size, doc_len, 2 * n_hidden], [batch_size, doc_len, n_class]
        '''
        x_context, hidden_states = self.pos_bilstm(x.float())
        # x_context is of shape (batch_size, max_doc_len, 2 * n_hidden)
        x = x_context.reshape(-1, 2 * self.n_hidden)
        x = self.dropout2(x)
        # x is of shape (batch_size * max_doc_len, 2 * n_hidden)
        pred_pos = F.softmax(self.pos_linear(x), dim = -1)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        pred_pos = pred_pos.reshape(-1, self.doc_len, self.n_class)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        return x_context, pred_pos

    def get_cause_prediction(self, x):
        '''
        input shape: [batch_size, doc_len, 2 * n_hidden + n_class]
        output(s) shape: [batch_size, doc_len, 2 * n_hidden], [batch_size, doc_len, n_class]
        '''
        x_context, hidden_states = self.cause_bilstm(x.float())
        # x_context is of shape (batch_size, max_doc_len, 2 * n_hidden)
        x = x_context.reshape(-1, 2 * self.n_hidden)
        x = self.dropout2(x)
        # x is of shape (batch_size * max_doc_len, 2 * n_hidden)
        pred_cause = F.softmax(self.cause_linear(x), dim = -1)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        pred_cause = pred_cause.reshape(-1, self.doc_len, self.n_class)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        return x_context, pred_cause

    def get_pair_prediction(self, x1, x2, distance):
        '''
        input(s) shape: [batch_size * doc_len, 2 * n_hidden], [batch_size * doc_len, 2 * n_hidden], 
                        [batch_size, doc_len * doc_len, embedding_dim_pos] 
        output shape: [batch_size, doc_len * doc_len, n_class]
        '''        
        x = create_pairs(x1, x2)
        # x is of shape (batch_size, max_doc_len * max_doc_len, 4 * n_hidden)
        x_distance = torch.cat([x, distance.float()], -1)
        # x_distance is of shape (batch_size, max_doc_len * max_doc_len, 4 * n_hidden + embedding_dim_pos)
        x_distance = x_distance.reshape(-1, 4 * self.n_hidden + self.embedding_dim_pos)
        x_distance = self.dropout3(x_distance)
        # x is of shape (batch_size * max_doc_len * max_doc_len, 4 * n_hidden + embedding_dim_pos)
        pred_pair = F.softmax(self.pair_linear2(self.relu(self.pair_linear1(x_distance))), dim = -1)
        # pred_pair is of shape (batch_size * max_doc_len * max_doc_len, n_class)
        pred_pair = pred_pair.reshape(-1, self.doc_len * self.doc_len, self.n_class)
        # pred_pair is of shape (batch_size, max_doc_len * max_doc_len, n_class)
        return pred_pair

    def forward(self, x, distance):
        '''
        input(s) shape: [batch_size, doc_len, sen_len, embedding_dim], 
                        [batch_size, doc_len * doc_len, embedding_dim_pos]
        output(s) shape: [batch_size, doc_len, n_class], [batch_size, doc_len, n_class], 
                         [batch_size, doc_len * doc_len, n_class]
        '''
        s = self.get_clause_embedding(x)
        x_pos, pred_pos = self.get_emotion_prediction(s)
        s_pred_pos = torch.cat([s, pred_pos], 2)
        x_cause, pred_cause = self.get_cause_prediction(s_pred_pos)
        pred_pair = self.get_pair_prediction(x_pos, x_cause, distance)
        return pred_pos, pred_cause, pred_pair

############################################ TRAIN #####################################################
def train_and_eval(Model, pos_cause_criterion, pair_criterion, optimizer):
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(
        embedding_dim, embedding_dim_pos, train_file_path, w2v_file)
    word_embedding = torch.from_numpy(word_embedding)
    # Train distance embeddings
    pos_embedding = torch.autograd.Variable(torch.from_numpy(pos_embedding))
    pos_embedding.requires_grad_(True)
    torch.save(word_embedding, './save/word_embedding.pth')
    torch.save(word_id_mapping, './save/word_id_mapping.pth')
    acc_cause_list, p_cause_list, r_cause_list, f1_cause_list = [], [], [], []
    acc_pos_list, p_pos_list, r_pos_list, f1_pos_list = [], [], [], []
    acc_pair_list, p_pair_list, r_pair_list, f1_pair_list = [], [], [], []
    #################################### LOOP OVER FOLDS ####################################
    for fold in range(1, 11):
        print('############# fold {} begin ###############'.format(fold))
        ############################# RE-INITIALIZE MODEL PARAMETERS #############################
        for layer in Model.parameters():
            nn.init.uniform_(layer.data, -0.10, 0.10)
        #################################### TRAIN/TEST DATA ####################################
        train_file_name = 'fold{}_train.txt'.format(fold)
        val_file_name = 'fold{}_val.txt'.format(fold)
        tr_y_position, tr_y_cause, tr_y_pair, tr_x, tr_sen_len, tr_doc_len, tr_distance = load_data_pair(
                        './data_combine_eng/'+train_file_name, word_id_mapping, max_doc_len, max_sen_len)
        val_y_position, val_y_cause, val_y_pair, val_x, val_sen_len, val_doc_len, val_distance = \
            load_data_pair('./data_combine_eng/'+val_file_name, word_id_mapping, max_doc_len, max_sen_len)
        max_f1_cause, max_f1_pos, max_f1_pair, max_f1_avg = [-1.] * 4
        #################################### LOOP OVER EPOCHS ####################################
        for epoch in range(1, training_epochs + 1):
            step = 1
            #################################### GET BATCH DATA ####################################
            for train, _ in get_batch_data_pair(
                tr_x, tr_sen_len, tr_doc_len, tr_y_position, tr_y_cause, tr_y_pair, tr_distance, batch_size):
                tr_x_batch, tr_sen_len_batch, tr_doc_len_batch, tr_true_y_pos, tr_true_y_cause, \
                tr_true_y_pair, tr_distance_batch = train
                Model.train()
                tr_pred_y_pos, tr_pred_y_cause, tr_pred_y_pair = Model(embedding_lookup(word_embedding, \
                tr_x_batch), embedding_lookup(pos_embedding, tr_distance_batch))
                ############################## LOSS FUNCTION AND OPTIMIZATION ##############################
                loss = pos_cause_criterion(tr_true_y_pos, tr_pred_y_pos, tr_doc_len_batch)*pos + \
                pos_cause_criterion(tr_true_y_cause, tr_pred_y_cause, tr_doc_len_batch)*cause + \
                pair_criterion(tr_true_y_pair, tr_pred_y_pair, tr_doc_len_batch)*pair
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #################################### PRINT AFTER EPOCHS ####################################
                if step % 25 == 0:
                    # print(Model.pair_linear.weight.shape); print(Model.pair_linear.weight.grad)
                    print('Fold {}, Epoch {}, step {}: train loss {:.4f} '.format(fold, epoch, step, loss))
                    acc, p, r, f1 = acc_prf_aux(tr_pred_y_pos, tr_true_y_pos, tr_doc_len_batch)
                    print('emotion_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 score {:.4f}'.format(
                            acc, p, r, f1))
                    acc, p, r, f1 = acc_prf_aux(tr_pred_y_cause, tr_true_y_cause, tr_doc_len_batch)
                    print('cause_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 score {:.4f}'.format(
                            acc, p, r, f1))
                    acc, p, r, f1 = acc_prf_pair(tr_pred_y_pair, tr_true_y_pair, tr_doc_len_batch)
                    print('pair_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 score {:.4f}'.format(
                            acc, p, r, f1)) 
                step += 1
            #################################### TEST ON 1 FOLD ####################################
            with torch.no_grad():
                Model.eval()
                val_pred_y_pos, val_pred_y_cause, val_pred_y_pair = Model(embedding_lookup(word_embedding, \
                val_x), embedding_lookup(pos_embedding, val_distance))

                loss = pos_cause_criterion(val_y_position, val_pred_y_pos, val_doc_len)*pos + \
                pos_cause_criterion(val_y_cause, val_pred_y_cause, val_doc_len)*cause + \
                pair_criterion(val_y_pair, val_pred_y_pair, val_doc_len)*pair
                print('Fold {} val loss {:.4f}'.format(fold, loss))
                acc, p, r, f1 = acc_prf_aux(val_pred_y_pos, val_y_position, val_doc_len)
                result_avg_pos = [acc, p, r, f1]
                if f1 > max_f1_pos:
                    max_acc_pos, max_p_pos, max_r_pos, max_f1_pos = acc, p, r, f1
                print('emotion_predict: val acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(
                    max_acc_pos, max_p_pos, max_r_pos, max_f1_pos))

                acc, p, r, f1 = acc_prf_aux(val_pred_y_cause, val_y_cause, val_doc_len)
                result_avg_cause = [acc, p, r, f1]
                if f1 > max_f1_cause:
                    max_acc_cause, max_p_cause, max_r_cause, max_f1_cause = acc, p, r, f1
                print('cause_predict: val acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(
                    max_acc_cause, max_p_cause, max_r_cause, max_f1_cause))

                acc, p, r, f1 = acc_prf_pair(val_pred_y_pair, val_y_pair, val_doc_len)
                result_avg_pair = [acc, p, r, f1]
                if f1 > max_f1_pair:
                    max_acc_pair, max_p_pair, max_r_pair, max_f1_pair = acc, p, r, f1
                print('pair_predict: val acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(
                    max_acc_pair, max_p_pair, max_r_pair, max_f1_pair))

            #################################### STORE BETTER PAIR F1 ####################################
            if result_avg_pair[-1] > max_f1_avg:
                torch.save(pos_embedding, "./save/pos_embedding_fold_{}.pth".format(fold))
                torch.save(Model.state_dict(), "./save/E2E-PextE_fold_{}.pth".format(fold))
                max_f1_avg = result_avg_pair[-1]
                result_avg_cause_max = result_avg_cause
                result_avg_pos_max = result_avg_pos
                result_avg_pair_max = result_avg_pair

            print('avg max cause: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(
                result_avg_cause_max[0], result_avg_cause_max[1], result_avg_cause_max[2], result_avg_cause_max[3]))
            print('avg max pos: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(
                result_avg_pos_max[0], result_avg_pos_max[1], result_avg_pos_max[2], result_avg_pos_max[3]))
            print('avg max pair: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(
                result_avg_pair_max[0], result_avg_pair_max[1], result_avg_pair_max[2], result_avg_pair_max[3]))

        print('############# fold {} end ###############'.format(fold))
        acc_cause_list.append(result_avg_cause_max[0])
        p_cause_list.append(result_avg_cause_max[1])
        r_cause_list.append(result_avg_cause_max[2])
        f1_cause_list.append(result_avg_cause_max[3])
        acc_pos_list.append(result_avg_pos_max[0])
        p_pos_list.append(result_avg_pos_max[1])
        r_pos_list.append(result_avg_pos_max[2])
        f1_pos_list.append(result_avg_pos_max[3])
        acc_pair_list.append(result_avg_pair_max[0])
        p_pair_list.append(result_avg_pair_max[1])
        r_pair_list.append(result_avg_pair_max[2])
        f1_pair_list.append(result_avg_pair_max[3])

    #################################### FINAL TEST RESULTS ON 10 FOLDS ####################################
    all_results = [acc_cause_list, p_cause_list, r_cause_list, f1_cause_list, \
    acc_pos_list, p_pos_list, r_pos_list, f1_pos_list, acc_pair_list, p_pair_list, r_pair_list, f1_pair_list,]
    acc_cause, p_cause, r_cause, f1_cause, acc_pos, p_pos, r_pos, f1_pos, acc_pair, p_pair, r_pair, f1_pair = \
        map(lambda x: np.array(x).mean(), all_results)
    print('\ncause_predict: val f1 in 10 fold: {}'.format(np.array(f1_cause_list).reshape(-1,1)))
    print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_cause, p_cause, r_cause, f1_cause))
    print('emotion_predict: val f1 in 10 fold: {}'.format(np.array(f1_pos_list).reshape(-1,1)))
    print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_pos, p_pos, r_pos, f1_pos))
    print('pair_predict: val f1 in 10 fold: {}'.format(np.array(f1_pair_list).reshape(-1,1)))
    print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_pair, p_pair, r_pair, f1_pair))

############################################### MAIN ########################################################
def main():
    Model = E2E_PextE(embedding_dim, embedding_dim_pos, max_sen_len, max_doc_len, \
    keep_prob1, keep_prob2, keep_prob3, n_hidden, n_class)
    Model.to(device)
    print(Model)
    x = torch.rand([batch_size, max_doc_len, max_sen_len, embedding_dim]).to(device)
    distance = torch.rand([batch_size, max_doc_len * max_doc_len, embedding_dim_pos]).to(device)
    pred_pos, pred_cause, pred_pair = Model(x, distance)
    print("Random i/o shapes x: {}, distance: {}, y_pos: {}, y_cause: {}, y_pair: {}".format(
        x.shape, distance.shape, pred_pos.shape, pred_cause.shape, pred_pair.shape))
    pos_cause_criterion = ce_loss_aux(); pair_criterion = ce_loss_pair(diminish_factor)
    optimizer = optim.Adam(Model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    train_and_eval(Model, pos_cause_criterion, pair_criterion, optimizer)

if __name__ == "__main__":
    main()
