# !/usr/bin/env bash
## created by Yun Hao @FunctionLab 2023
## This script contains command line example that fine-tune a pre-trained sequence-based transfer learning model with Seal framework, which predicts gene expression in specific biological contexts

# Usage: 
# python seal_train.py --general_exp_file <general context expression file> --pretrained <True> --pretrained_name <pre-trained file location> --n_latent <number of hidden neurons> --lr_pretrain <pre-training learning rate> --l2_pretrain <pre-training L2 regularization factor> --general_context_group <general context group info file> --spec_gene_weight <weight assigned to specific genes> --finetune_exp_file <specific context expression file> --lr_finetune <fine-tuning learning rate> --l2_finetune <fine-tuning L2 regularization factor> --specific_context_group <specific context group info file> --out_name <output file location>

# Load the pre-trained mnodel predicting expression of generic adult brain samples, fine-tune towards scRNA-seq expression data of brain development (Herring et al Cell 2022)
# run './test/train/test_train_from_scratch.sh' first before running the command below
python seal_train.py --general_exp_file resource/geneanno.exp.csv_general_brain.tsv --pretrained True --pretrained_name test/train/brain_dev_exp_tissue_state --n_latent 1000 --spec_gene_weight 2 --finetune_exp_file resource/brain_dev_exp_cell_state1.tsv --lr_finetune 0.0001 --l2_finetune 0.0001 --specific_context_group resource/brain_dev_exp_cell_group1.tsv --out_name test/train/brain_dev_exp_cell_state1
