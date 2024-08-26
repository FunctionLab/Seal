# !/usr/bin/env python
## created by Yun Hao @FunctionLab 2023
## This script contains functions to process expression data


## Module
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


## This function processes the expression matrix data: filter genes, add pseudo counts, transform by log, split into train/test set 
def process_expression_data(gene_exp_df, anno_file, gene_type = 'all', exclude_chromo = ['chrX', 'chrY'], test_chromo = ['chr8', 'chr9']):
    ## 0. Input arguments
        # gene_exp_df (pandas DataFrame): data frame containing the context-specific expression of genes (row: gene; column: context)
        # anno_file (str): name of file containing the gene annotation file  
        # gene_type (str): type of gene whose expression data will be used to train model ('all', 'pc', 'lincRNA', 'rRNA') 
        # exclude_chromo (list): list of chromosomes of which associated genes are excluded from the model training/test
        # test_chromo (list): list of chromosomes of which associated genes are included in the testing set

    ## 1 Filter genes by specified type and chromosome     
    gene_anno_df = pd.read_csv(anno_file)
    if gene_type == 'pc':
        gene_id = np.asarray(gene_anno_df.type == 'protein_coding')
    elif gene_type == 'lincRNA':
        gene_id = np.asarray(gene_anno_df.type == 'lincRNA')
    else:
        gene_id = np.asarray(gene_anno_df.type != 'rRNA')
    auto_chro_id = np.asarray([not gads in exclude_chromo for gads in gene_anno_df.seqnames])

    ## 2. Add pseudo counts to gene expression and make log transformation  
    gene_anno_exp_df = pd.merge(gene_anno_df, gene_exp_df, how = 'left', left_on = 'id', right_on = 'gene_id')
    N_cols = gene_anno_exp_df.shape[1]
    # Pseudo count
    ga_exp_df = gene_anno_exp_df.iloc[:, 8:N_cols]
    ged_min = np.log10(ga_exp_df[ga_exp_df > 0].min().values)
    pseudocounts = np.power(10, np.floor(ged_min) + 1) 
    # log transformation
    exp_df = np.log(ga_exp_df + pseudocounts)
    null_id = np.asarray(exp_df.sum(axis = 1, skipna = False).isnull().values)
    valid_id = np.invert(null_id)

    ## 3. Split genes into training/testing set based on specified chromosomes
    # training set 
    train_chro_id = np.asarray([not gads in test_chromo for gads in gene_anno_df.seqnames])
    train_id = gene_id * auto_chro_id * valid_id * train_chro_id
    train_exp = exp_df[train_id].values
    # testing set
    test_chro_id = np.asarray([gads in test_chromo for gads in gene_anno_df.seqnames]) 
    test_id = gene_id * auto_chro_id * valid_id * test_chro_id
    test_exp = exp_df[test_id].values
    
    return train_exp, test_exp, train_id, test_id


## This function defines formats feature-response array data into tensors used for MLP model training  
class exp_dataformat(Dataset):
    ## 0. Input arguments 
        # feature_data: array that contains input feature data 
        # label_data: array that contains input label/response data

    ## 1. Convert feature and label arrays into tensors 
    def __init__(self, feature_data, label_data, weight_data):
        super(exp_dataformat, self).__init__()
        self.features = torch.tensor(feature_data, dtype = torch.float)
        labels = torch.tensor(label_data, dtype = torch.float)
        weights = torch.tensor(weight_data, dtype = torch.float)
        if len(labels.shape) == 1:
            self.labels = labels.view(labels.shape[0], 1)
            self.weights = weights.view(weights.shape[0], 1)
        else:
            self.labels = labels
            self.weights = weights

    ## 2. Get feature and label data by index
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        weight = self.weights[index]
        return feature, label, weight

    ## 3. Obtain number of data samples 
    def __len__(self):
        return len(self.labels)


