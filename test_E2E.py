############################################ IMPORT ##########################################################
import sys, os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from utils.funcs import *
from utils.prepare_data import *
from E2E_PextE import *
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
keep_prob1 = 0.8                                                    # word embedding training dropout keep prob
keep_prob2 = 1.0                                                    # softmax layer dropout keep prob
keep_prob3 = 1.0                                                    # softmax layer dropout keep prob

################################################ TEST #########################################################
def test(Model):
    word_embedding = torch.load("./save/word_embedding.pth")
    word_id_mapping = torch.load("./save/word_id_mapping.pth")
    acc_cause_list, p_cause_list, r_cause_list, f1_cause_list = [], [], [], []
    acc_pos_list, p_pos_list, r_pos_list, f1_pos_list = [], [], [], []
    acc_pair_list, p_pair_list, r_pair_list, f1_pair_list = [], [], [], []
    #################################### LOOP OVER FOLDS ####################################
    for fold in range(1, 11):
        print('############# fold {} begin ###############'.format(fold))
        #################################### LOAD TEST DATA ####################################
        test_file_name = 'fold{}_test.txt'.format(fold)
        te_y_position, te_y_cause, te_y_pair, te_x, te_sen_len, te_doc_len, te_distance = load_data_pair(
            './data_combine_eng/'+test_file_name, word_id_mapping, max_doc_len, max_sen_len)
        pos_embedding = torch.load("./save/pos_embedding_fold_{}.pth".format(fold))
        Model.load_state_dict(torch.load("./save/E2E-PextE_fold_{}.pth".format(fold)))
        with torch.no_grad():
            Model.eval()
            te_pred_y_pos, te_pred_y_cause, te_pred_y_pair = Model(embedding_lookup(word_embedding, \
                te_x), embedding_lookup(pos_embedding, te_distance))
            # emotion results
            acc, p, r, f1 = acc_prf_1st_step(te_pred_y_pos, te_y_position, te_doc_len)
            acc_pos_list.append(acc); p_pos_list.append(p); r_pos_list.append(r); f1_pos_list.append(f1)
            print("Fold {} emotion acc: {:.4f} p: {:.4f} r: {:.4f} f1: {:.4f}".format(fold, acc, p, r, f1))
            # cause results
            acc, p, r, f1 = acc_prf_1st_step(te_pred_y_cause, te_y_cause, te_doc_len)
            acc_cause_list.append(acc); p_cause_list.append(p); r_cause_list.append(r); f1_cause_list.append(f1)
            print("Fold {} cause acc: {:.4f} p: {:.4f} r: {:.4f} f1: {:.4f}".format(fold, acc, p, r, f1))
            # pair results
            acc, p, r, f1 = acc_prf_pair(te_pred_y_pair, te_y_pair, te_doc_len)
            acc_pair_list.append(acc); p_pair_list.append(p); r_pair_list.append(r); f1_pair_list.append(f1)
            print("Fold {} pair acc: {:.4f} p: {:.4f} r: {:.4f} f1:{:.4f}".format(fold, acc, p, r, f1))

        print('############# fold {} end ###############\n'.format(fold))

    #################################### FINAL TEST RESULTS ON 10 FOLDS ####################################
    all_results = [acc_cause_list, p_cause_list, r_cause_list, f1_cause_list, \
    acc_pos_list, p_pos_list, r_pos_list, f1_pos_list, acc_pair_list, p_pair_list, r_pair_list, f1_pair_list,]
    acc_cause, p_cause, r_cause, f1_cause, acc_pos, p_pos, r_pos, f1_pos, acc_pair, p_pair, r_pair, f1_pair = \
        map(lambda x: np.array(x).mean(), all_results)
    print('\ncause_predict: test f1 in 10 fold: {}'.format(np.array(f1_cause_list).reshape(-1,1)))
    print('\naverage : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc_cause, p_cause, r_cause, f1_cause))
    print('\nemotion_predict: test f1 in 10 fold: {}'.format(np.array(f1_pos_list).reshape(-1,1)))
    print('\naverage : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc_pos, p_pos, r_pos, f1_pos))
    print('\npair_predict: test f1 in 10 fold: {}'.format(np.array(f1_pair_list).reshape(-1,1)))
    print('\naverage : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc_pair, p_pair, r_pair, f1_pair))

############################################### MAIN ########################################################
def main():
    Model = E2E_PextE(embedding_dim, embedding_dim_pos, max_sen_len, max_doc_len, \
    keep_prob1, keep_prob2, keep_prob3, n_hidden, n_class).to(device)
    test(Model)

if __name__ == "__main__":
    main()