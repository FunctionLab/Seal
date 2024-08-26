# !/usr/bin/env python
## created by Yun Hao @Function Lab 2023
## This script contains functions for Module 1 (predicting chromatin profile from sequence) and Module 2 (transforming predicted chromatin profile by exponential functions) of Seal framework.   


# Module
import sys
import numpy as np
import pandas as pd
import pyfasta
import torch
from torch import nn


## Concatenate 3d tensor to 1d 
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn
    
    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


## Data format switch from convolutional layer to fully connected layer 
class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


## DeepSEA BELUGA model architecture   
class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


## This function fetches genome sequence centered on the specific location of ref/alt allele
def fetch_sequence(genome, chrom, pos, ref, alt, shift, input_len):
    ## 0. Input arguments
        # genome (pyfasta Fasta): Fasta object of query genome
        # chrom (str): the chromosome name that must matches one of the names in CHRS.
        # pos (int): chromosome coordinate (1-based).
        # ref (str): the reference allele.
        # alt (str): the alternative allele.
        # shift (int): retrived sequence center position - variant position.
        # input_len (int): the targeted sequence length (input_len+100bp is retrived for reference allele).

    ## 1. Obtain the starting and ending coordinate of sequence centered on the ref/alt allele 
    window_len = input_len + 100
    start_pos = pos + shift - int(window_len / 2 - 1)
    end_pos = pos + shift + int(window_len / 2)
    f_seq = genome.sequence({'chr': chrom, 'start': start_pos, 'stop': end_pos})

    ## 2. Fetch sequence from fasta object
    mut_lower = int(window_len / 2 - 1 - shift)
    mut_upper = int(window_len / 2 - 1 - shift) + len(ref)
    ref_seq = f_seq[:mut_lower] + ref + f_seq[mut_upper:]
    alt_seq = f_seq[:mut_lower] + alt + f_seq[mut_upper:]
    ref_check = f_seq[mut_lower: mut_upper].upper() == ref.upper()
    
    return ref_seq, alt_seq, ref_check 


## This function encode sequence into input tensors for BELUGA  
def encode_sequence(seq_list, input_len, include_reverse = True):
    ## 0. Input arguments
        # seq_list (list): list containing the sequence to be encoded 
        # input_len (int): the targeted sequence length 
        # include_reverse (bool): whether to generate encoding for the reverse strand  

    ## 1. One-hot encoding of nucleotides 
    encode_dict = {
            'A': [1, 0, 0, 0],
            'a': [1, 0, 0, 0],
            'G': [0, 1, 0, 0],
            'g': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'c': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            't': [0, 0, 0, 1],
            'N': [0, 0, 0, 0],
            'n': [0, 0, 0, 0],
            'H': [0, 0, 0, 0],
            '-': [0, 0, 0, 0]
            }
    
    ## 2. Convert forward strand sequence to encoding based on dictionary  
    encode_list = np.zeros((len(seq_list), 4, input_len), np.bool_)
    for isl, sl in enumerate(seq_list):
        lsl = len(sl)
        lower_id = int(np.floor((lsl - input_len) / 2.0))
        upper_id = int(np.floor(lsl - (lsl - input_len) / 2.0))
        cut_sl = sl[lower_id: upper_id]
        encode_list[isl, :, :] = np.array([encode_dict[csl] for csl in cut_sl]).T
   
    ## 3. Convert reverse strand sequence to encoding based on dictionary 
    if include_reverse == True:
        reverse_encode_list = encode_list[:, ::-1, ::-1].copy()
        return torch.tensor(encode_list.astype(np.float32)), torch.tensor(reverse_encode_list.astype(np.float32))
    else:
        return torch.tensor(encode_list.astype(np.float32))


## This function predicts the chromatin profile of ref/alt allele centered sequences (+/- shift) for each variant   
def predict_variant_chromatin_effect(var_df, fasta_file, shift, input_len, model_file, feat_size, batch_size):
    ## 0. Input arguments
        # var_df (pandas DataFrame): data frame containing variant info (see var_data)
        # fasta_file (str): name of genome fasta file
        # shift (int): retrived sequence center position - variant position.
        # input_len (int): the targeted sequence length
        # model_file (str): name of trained BELUGA model file
        # feat_size (int): number of chromatin features BELUGA predicts
        # batch_size (int): batch size for running BELUGA

    ## 1. Fetch genome sequence around the ref/alt allele of each variant 
    genome = pyfasta.Fasta(fasta_file)
    N_var = var_df.shape[0]
    ref_seqs = [''] * N_var
    alt_seqs = [''] * N_var
    ref_match = np.zeros(N_var, np.bool_)
    for rnv in range(0, N_var):
        rnv_row = var_df.iloc[rnv, :]
        ref_seqs[rnv], alt_seqs[rnv], ref_match[rnv] = fetch_sequence(genome, rnv_row['chrom'], rnv_row['pos'], rnv_row['ref'], rnv_row['alt'], shift, input_len)

    ## 2. Encode the ref and alt sequence into input tensors for BELUGA 
    ref_feat1, ref_feat2 = encode_sequence(ref_seqs, input_len)
    alt_feat1, alt_feat2 = encode_sequence(alt_seqs, input_len)

    ## 3. Predict chromatin profile of input ref and alt sequences
    # load BELUGA model
    model = Beluga()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    # predict with BELUGA
    ref_pred = np.zeros((N_var, feat_size), np.float32)
    alt_pred = np.zeros((N_var, feat_size), np.float32)
    batch_id = np.arange(0, N_var, batch_size) 
    batch_id = np.insert(batch_id, len(batch_id), N_var)
    Nbi = len(batch_id)
    for ibi in range(0, Nbi - 1):
        ibi_lower = batch_id[ibi] 
        ibi_upper = batch_id[ibi + 1]
        # ref sequence (score = (forward + reverse)/2)
        ibi_ref1 = model.forward(ref_feat1[ibi_lower:ibi_upper, :, :].unsqueeze(2)).detach().numpy().copy()
        ibi_ref2 = model.forward(ref_feat2[ibi_lower:ibi_upper, :, :].unsqueeze(2)).detach().numpy().copy()
        ref_pred[ibi_lower:ibi_upper, :] = (ibi_ref1 + ibi_ref2)/2
        # alt sequence 
        alt_ref1 = model.forward(alt_feat1[ibi_lower:ibi_upper, :, :].unsqueeze(2)).detach().numpy().copy()
        alt_ref2 = model.forward(alt_feat2[ibi_lower:ibi_upper, :, :].unsqueeze(2)).detach().numpy().copy()
        alt_pred[ibi_lower:ibi_upper, :] = (alt_ref1 + alt_ref2)/2

    return ref_pred, alt_pred, ref_match


## This function computes the exponential weights of sliding windows for transforming predicted chromatin profile 
def compute_exp_weight(bin_loc, coeff = [0.01, 0.02, 0.05, 0.1, 0.2]):
    ## 0. Input argument
        # bin_loc (numpy array): array containing the relative distance between gene TSS and center of each sliding window
        # coeff: coefficient of the exponential functions to be used in transformation 

    coeff1 = np.broadcast_to(coeff, (len(bin_loc), len(coeff))).T
    bin_loc1 = np.broadcast_to(bin_loc, (len(coeff), len(bin_loc)))
    # compute the weight separately for windows upstream and downstream of TSS, as the chromatin profile will also be transformed separately  
    down_wei = np.exp(-coeff1 * np.abs(bin_loc1)/200) * (bin_loc1 <= 0)
    up_wei = np.exp(-coeff1 * np.abs(bin_loc1)/200) * (bin_loc1 >= 0)
    all_wei = np.concatenate([down_wei, up_wei], axis = 0)

    return all_wei


## This function transforming predicted chromatin profile of sliding windows along TSS-proximal region by exponential functions 
def transform_chromatin_by_exp(ref_pred, alt_pred, shift, var_df, gene_row_dict, feat_size, bin_loc = np.arange(-19900, 20100, 200), gene_feature_folder = 'resource/chromatin_tss/'):
    ## 0. Input arguments
        # ref_pred (numpy array): predicted chromatin profile of variant ref alleles  
        # alt_pred (numpy array): predicted chromatin profile of variant alt alleles  
        # shift (int): retrived sequence center position - variant position
        # var_df (pandas DataFrame): data frame containing variant info (see var_data)
        # gene_row_dict (dictionary): dictionary containing the var_df row indices that are matched to each gene 
        # feat_size (int): number of chromatin features BELUGA predicts 
        # bin_loc (numpy array): array containing the relative distance between gene TSS and center of each sliding window
        # gene_feature_folder (str): name of folder containing pre-computed chromatin profile of TSS-proximal sequence of each gene using reference genome 

    ## 1. Iterate by variants by the matched gene   
    N_var = var_df.shape[0]
    N_exp_feat = feat_size * 10
    ref_exp_feat = np.zeros((N_var, N_exp_feat), np.float32)
    alt_exp_feat = np.zeros((N_var, N_exp_feat), np.float32)
    for gene, row_id in gene_row_dict.items():
        # load the pre-computed chromatin profile of TSS-proximal sequence of current gene using reference genome 
        gene_feat = np.load(gene_feature_folder + gene + '.npy')
        # compute the distance between variant (+/- shift) and TSS of current gene
        tss_dist = (var_df.dist_to_tss.values[row_id] + shift) * ((var_df.gene_strand.values[row_id] == '+') * 2 - 1)
        # iterate by variant  
        for iri, ri in enumerate(row_id): 
            # obtain the index of variant-centered sequence among the sliding windows around TSS-proximal region  
            iri_dist_id = int(np.ceil(tss_dist[iri]/200) + 99)
            if iri_dist_id < 0:
                iri_dist_id = 0
            if iri_dist_id > 199:
                iri_dist_id = 199
            # substitue the center of the sliding window with the variant coordinate (+/- shift) 
            iri_bin_loc = bin_loc.copy()
            iri_bin_loc[iri_dist_id] = tss_dist[iri]
            # compute exponential weight for transformation 
            iri_bin_wei = compute_exp_weight(iri_bin_loc)
            # substitute the pre-computed profile of reference genome with predicted profile of reference allele, transform newly predicted chromatin profile  
            iri_ref_feat = gene_feat.copy()
            iri_ref_feat[iri_dist_id, :] = ref_pred[ri, :]
            ref_exp_feat[ri, :] = np.dot(iri_bin_wei, iri_ref_feat).reshape(N_exp_feat)
            # substitute the pre-computed profile of reference genome with predicted profile of alternative allele, transform newly predicted chromatin profile 
            iri_alt_feat = gene_feat.copy()
            iri_alt_feat[iri_dist_id, :] = alt_pred[ri, :]
            alt_exp_feat[ri, :] = np.dot(iri_bin_wei, iri_alt_feat).reshape(N_exp_feat)

    return ref_exp_feat, alt_exp_feat

