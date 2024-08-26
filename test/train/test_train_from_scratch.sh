# !/usr/bin/env bash
## created by Yun Hao @FunctionLab 2023
## This script contains command line example that trains a sequence-based transfer learning model with Seal framework from scratch, which predicts gene expression in specific biological contexts

# Usage: 
# python seal_train.py --general_exp_file <general context expression file> --n_latent <number of hidden neurons> --lr_pretrain <pre-training learning rate> --l2_pretrain <pre-training L2 regularization factor> --general_context_group <general context group info file> --spec_gene_weight <weight assigned to specific genes> --finetune_exp_file <specific context expression file> --lr_finetune <fine-tuning learning rate> --l2_finetune <fine-tuning L2 regularization factor> --specific_context_group <specific context group info file> --out_name <output file location>

# Pre-train on GTEx/ENCODE/Roadmap Bulk RNA-seq expression data of generic adult brain samples, then fine-tune towards PsyCHENCODE RNA-seq expression data of brain development 
python seal_train.py --general_exp_file resource/geneanno.exp.csv_general_brain.tsv --n_latent 1000 --lr_pretrain 0.001 --l2_pretrain 1e-05 --spec_gene_weight 2 --finetune_exp_file resource/brain_dev_exp_tissue_state.tsv --lr_finetune 1e-06 --l2_finetune 0.01 --specific_context_group resource/brain_dev_exp_tissue_group.tsv --out_name test/train/brain_dev_exp_tissue_state

