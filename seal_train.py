# !/usr/bin/env python
## created by Yun Hao @FunctionLab 2023
## This script trains the Seal framework, a transfer learning model for predicting gene expression of specific contexts 


## Module
import sys
import argparse
import torch
import numpy as np
import pandas as pd
sys.path.insert(0, 'function/')
import exp_data
import exp_effect
torch.set_num_threads(20)


## 0. Inputs for training Seal model predicting gene expression from chromatin profile 
parser = argparse.ArgumentParser(description = 'Process some integers.')
# input for pre-training 
parser.add_argument('--general_exp_file', action = 'store', dest = 'general_exp_file', type = str)
parser.add_argument('--gene_anno_file', action = 'store', dest = 'gene_anno_file', type = str, default = 'resource/geneanno.csv')
parser.add_argument('--pretrained', action = 'store', dest = 'pretrained', type = bool, default = False)
parser.add_argument('--pretrained_name', action = 'store', dest = 'pretrained_name', type = str, default = 'NA')
parser.add_argument('--n_latent', action = 'store', dest = 'n_latent', type = int)
parser.add_argument('--lr_pretrain', action = 'store', dest = 'lr_pretrain', type = float, default = 1e-4)
parser.add_argument('--l2_pretrain', action = 'store', dest = 'l2_pretrain', type = float, default = 1e-5)
parser.add_argument('--general_context_group', action = 'store', dest = 'general_context_group', type = str, default = 'NA')
parser.add_argument('--spec_gene_weight', action = 'store', dest = 'spec_gene_weight', type = float, default = 1.0)
# input for fine-tuning
parser.add_argument('--finetune_exp_file', action = 'store', dest = 'finetune_exp_file', type = str)
parser.add_argument('--lr_finetune', action = 'store', dest = 'lr_finetune', type = float)
parser.add_argument('--l2_finetune', action = 'store', dest = 'l2_finetune', type = float)
parser.add_argument('--specific_context_group', action = 'store', dest = 'specific_context_group', type = str, default = 'NA')
parser.add_argument('--out_name', action = 'store', dest = 'out_name', type = str)
args = parser.parse_args()

## 1. Pre-train a neural network model for predicting gene expression of general contexts 
# read in and process gene expression data of general context, split into training/testing label array 
pt_exp_df = pd.read_csv(args.general_exp_file, sep = '\t')
pt_train_res, pt_test_res, pt_train_id, pt_test_id = exp_data.process_expression_data(pt_exp_df, args.gene_anno_file)
# read in TSS-proximal chromatin profile of genes, split into training/testing feature matrix
epi_feat_data = np.load('resource/all_gene_chromatin_exp.npy')
pt_train_feat, pt_test_feat = epi_feat_data[pt_train_id], epi_feat_data[pt_test_id]
# name output files of pre-training 
out_folder = args.out_name + '_seal'
# pre-training with expression data of general context
if args.pretrained:
    # if pre-trained model already exist, load the pre-trained model
    at_folder = args.pretrained_name + '_seal'
    pt_model, pt_eval_perf, pt_valid_loss = exp_effect.load_pretrained_model(pt_train_feat, pt_train_res, args.n_latent, at_folder, out_folder)
else:
    # otherwise pre-train a neural network model predicting gene expression of general context from chromatin profile 
    torch.manual_seed(0)
    pt_model, pt_train_sum, pt_valid_loss = exp_effect.train_tl_model(pt_train_feat, pt_train_res, 'pre-train', 
            N_latent = args.n_latent, 
            model_name = out_folder + '_pt_model.pt', 
            learning_rate = args.lr_pretrain, 
            l2_lambda = args.l2_pretrain,
            group_file = args.general_context_group,
            spec_rate = args.spec_gene_weight)
    pt_train_sum.to_csv(out_folder + '_pt_training_loss_summary.tsv', sep = '\t', index = False, float_format = '%.5f')
    # save performance evaluation result of pre-trained model on the testing set
    pt_eval_perf = exp_effect.evaluate_tl_model(pt_model, pt_test_feat, pt_test_res)
    pt_eval_perf.index = pt_exp_df.columns[1: ]
    pt_eval_perf.to_csv(out_folder + '_pt_testing_perf.tsv', sep = '\t', float_format = '%.5f')

## 2. Fine-tune pre-trained model towards predicting gene expression of specific contexts 
# read in and process gene expression data of specific context, split into training/testing label array 
ft_exp_df = pd.read_csv(args.finetune_exp_file, sep = '\t') 
ft_train_res, ft_test_res, ft_train_id, ft_test_id = exp_data.process_expression_data(ft_exp_df, args.gene_anno_file)
ft_train_feat, ft_test_feat = epi_feat_data[ft_train_id], epi_feat_data[ft_test_id]
# fine-tune with expression data of specific context  
torch.manual_seed(0)
ft_model, ft_train_sum, ft_valid_loss = exp_effect.train_tl_model(ft_train_feat, ft_train_res, 'fine-tune', 
        N_latent = args.n_latent, 
        learned_model = pt_model, 
        model_name = out_folder + '_ft_model.pt', 
        learning_rate = args.lr_finetune, 
        l2_lambda = args.l2_finetune,
        group_file = args.specific_context_group,
        spec_rate = args.spec_gene_weight)
ft_train_sum.to_csv(out_folder + '_ft_training_loss_summary.tsv', sep = '\t', index = False, float_format = '%.5f')
# save performance evaluation result of fine-tuned model on the testing set   
ft_eval_perf = exp_effect.evaluate_tl_model(ft_model, ft_test_feat, ft_test_res)
ft_eval_perf.index = ft_exp_df.columns[1: ]
ft_eval_perf.to_csv(out_folder + '_ft_testing_perf.tsv', sep = '\t', float_format = '%.5f')
# save performance of models on the validation set (for hyperparameter tuning)
valid_perf = pd.DataFrame([pt_valid_loss, ft_valid_loss])
valid_perf.index = ['pretrain_model_loss', 'finetune_model_loss']
valid_perf.to_csv(out_folder + '_validation_perf.tsv', sep = '\t', header = False, float_format = '%.5f')
# save performance summary to output model summary file   
all_perf_summary = exp_effect.summarize_tl_model(out_folder, args.n_latent, pt_eval_perf, ft_eval_perf)
perf_sum_file = open(out_folder + '_model_summary.txt', 'w')
for aps in all_perf_summary:
    perf_sum_file.write('%s\n' % aps)
perf_sum_file.close()
