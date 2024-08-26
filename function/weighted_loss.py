# !/usr/bin/env python
## created by Yun Hao @FunctionLab 2023
## This script contains loss functions used in the training of Module 3 (neural network transfer learning model predicting gene expression) of Seal framework 


## Module
import numpy as np
import pandas as pd
import torch


## This function computes gene weights of loss function based gene expression variation across tissue contexts   
def compute_gene_weight(tissue_exp, tissue_group_file = 'NA', specificity_rate = 2, high_quantile = 0.7, low_quantile = 0.1):
    ## 0. Input arguments:
        # tissue_exp (numpy array): array containing the expression matrix of tissue contexts  
        # tissue_group_file (str): name of file containing group info of tissue contexts
        #$ specificity_rate (float): specificity rate for assigning weight to genes 
        # high_quantile (float): percentile threshold for highly specific genes (gene will be assigned higher weight if expression variation is higher than this threshold)
        # low_quantile (float): percentile threshold for highly specific genes (gene will be assigned lower weight if expression variation is lower than this threshold)

    ## 1. Compute the weight values to be assigned to highly/lowly specific genes
    high_w = specificity_rate
    low_w = 1/specificity_rate

    ## 2. Assign weights to genes based on their expression variation across tissue contexts 
    te_shape = tissue_exp.T.shape
    if tissue_group_file == 'NA':
        # if not group file is NOT specified, then assign weights based on the gene expression variation across all tissue contexts  
        tissue_sd = np.std(tissue_exp, axis = 1)
        high_sd = np.quantile(tissue_sd, high_quantile)
        low_sd = np.quantile(tissue_sd, low_quantile) 
        wei = np.ones(len(tissue_sd)) 
        wei = wei + (high_w - 1) * (tissue_sd > high_sd) + (low_w - 1) * (tissue_sd < low_sd)
        tissue_wei = np.broadcast_to(wei, te_shape).T
    else:
        # if not group file is specified, then assign weights based on the gene expression variation within each group 
        tissue_group_df = pd.read_csv(tissue_group_file, sep = '\t', header = None)
        tissue_wei = np.ones(tissue_exp.shape)
        for i in range(0, tissue_group_df.shape[0]):
            i_col = np.array(tissue_group_df.iloc[i, 1].split(','), dtype = int) - 1
            i_wei = np.ones(te_shape[1])
            # high/lowe weights assigned only applicable when the group has more than 1 context. Otherwise all genes share the same weight: 1
            if len(i_col) > 1:
                i_sd = np.std(tissue_exp[:, i_col], axis = 1)
                i_high = np.quantile(i_sd, high_quantile)
                i_low = np.quantile(i_sd, low_quantile)
                i_wei = np.ones(len(i_sd))
                i_wei = i_wei + (high_w - 1) * (i_sd > i_high) + (low_w - 1) * (i_sd < i_low)
            tissue_wei[:, i_col] = np.broadcast_to(i_wei, (len(i_col), len(i_sd))).T

    return tissue_wei


## This functionc computes weighted mean squared error loss function comparing predicted expresssion and observed expression of genes   
def weighted_mse_loss(pred, res, weight):
    ## 0. Input arguments:
        # pred (torch tensor): tensor containing predicted expression values of genes
        # res (torch tensor): tensor containing observed expression values of genes 
        # weight (torch tensor): tensor containing weight values of genes

    ## 1.
    weight_sum = torch.sum(weight, axis = 0)
    weight_sum1 = torch.broadcast_to(weight_sum, weight.shape)
    mse_sum = torch.sum(weight * (pred - res) ** 2/weight_sum1)
    mse = mse_sum/pred.shape[1]

    return mse


