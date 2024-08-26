# !/usr/bin/env python
## created by Yun Hao @FunctionLab 2023
## This script contains functions to process variant vcf data


## Module
import os
import numpy as np
import pandas as pd


## This function run closest-features from BEDOPS (https://bedops.readthedocs.io/en/latest/) to find the closest TSS of each variant 
def find_closest_tss(vcf_file, sort_bed_file):
    # 0. Input arguments
        # vcf_file (str): name of the variant vcf file 
        # sort_bed_file (str): name of the gene annotation bed file 
    
    ## 1. Create command running closest-features on the vcf file 
    find_file = vcf_file + '.bed.sorted.bed.closestgene'
    bed_command1 = 'closest-features --delim \'\\t\' --closest --dist <(awk \'{printf $1\"\\t\"$2-1\"\\t\"$2\"\\n\"}\' '
    bed_command2 = vcf_file + '|sed s/chr//g|sed s/^/chr/g|sort-bed - ) '
    bed_command3 = sort_bed_file + ' > ' + find_file
    bed_command = bed_command1 + bed_command2 + bed_command3
    
    ## 2. Write command file 
    command_file = 'find_closest_tss_' + vcf_file.split('/')[-1] + '.sh'
    f = open(command_file, 'w')
    f.write(bed_command)
    f.close()
    
    ## 3. Run the command, then delete the command file 
    os.system('chmod 775 ' + command_file)
    os.system('bash -c ./' + command_file)
    os.system('rm ' + command_file)
    
    return find_file


## This function processes the variant vcf data: finding closest TSS, filtering variants, grouping variants by closest gene
def process_variant_data(vcf_file, sort_bed_file):
    ## 0. Input arguments
        # vcf_file (str): name of the variant vcf file 
        # sort_bed_file (str): name of the gene sorted bed file (default value is the sorted bed file for hg19)

    ## 1. Read in variant vcf file 
    vcf_df = pd.read_csv(vcf_file, sep = '\t', header = None, comment = '#')
    vcf_df.iloc[:, 0] = 'chr' + vcf_df.iloc[:, 0].map(str).str.replace('chr', '')
    vcf_df = vcf_df.iloc[:, 0:5]
    vcf_df = vcf_df.drop_duplicates()

    ## 2. Find the closest gene TSS of each variant 
    closest_file = find_closest_tss(vcf_file, sort_bed_file)
    closest_df = pd.read_csv(closest_file, sep = '\t', header = None) 
    closest_df = closest_df.drop_duplicates()
    # merge info 
    vc_df = pd.merge(vcf_df, closest_df, left_on = [0, 1], right_on = [0, 2])
    vc_df1 = vc_df.iloc[:, [0, 1, 4, 5, 11, 12, 13]] 
    vc_df1.columns = ['chrom', 'pos', 'ref', 'alt', 'gene_strand', 'gene_id', 'dist_to_tss'] 

    ## 3. Filter variants 
    # check whether the chromosome info is corrent 
    chrs = ['chr' + str(i) for i in range(1, 23)]
    all_chrs = chrs + ['chrX', 'chrY']
    all_chrs_check = vc_df1.chrom.isin(all_chrs).values
    # check whether variant is within 20kb of TSS
    vc_df1['dist_to_tss'] = -vc_df1['dist_to_tss'].values
    tss_dist_check = (vc_df1.dist_to_tss.abs() <= 20000).values
    filter_id = all_chrs_check * tss_dist_check
    vc_df1 = vc_df1[filter_id]

    ## 4. Group variants by closest gene
    gene_row_dict = {}
    all_gene_names = vc_df1.gene_id.unique()
    for agn in all_gene_names:
        gene_row_dict[agn] = np.where(vc_df1.gene_id == agn)[0]

    return vc_df1, gene_row_dict

