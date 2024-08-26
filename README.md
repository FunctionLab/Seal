# *Seal*: *S*equence-to-*e*xpression tr*a*nsfer *l*earning framework for context-specific prediction of variant effects on expression 

**Yun Hao, Chandra L. Theesfeld, and Olga G. Troyanskaya**

**Flatiron Institute, Princeton University**

Seal is a framework for training sequence-based deep learning models that predict variant effect on gene expression in specific biological contexts with limited data. Seal adopts a transfer learning scheme that first pre-train on extensive-profiled expression data under general contexts, then fine-tune the model towards context-specific expression data. We applied the SEAL framework to train models predicting gene expression of brain development at both tissue and cell type resolution. In total, our trained Seal models can predict variant effects on gene expression under 122 tissue states and 680 cell states from early fetal to adulthood. This repository contains code for training sequence-to-expression models with Seal framework, and implementing the framework for variant effect prediction in tissue and cell states of brain development. 

The Seal framework is described in the following manuscript: [Link]()

## Setup 

### Requirements

Seal requires Python 3.6+ and Python packages PyTorch (>=1.9). Follow PyTorch installation steps [here](https://pytorch.org/). The other dependencies can be installed by running `pip install -r requirements.txt`. Seal also relies on the 'closest-features' from BEDOPS for finding the closest representative TSS of each variant. Follow installation steps [here](https://bedops.readthedocs.io/en/latest/).

### Install 

Clone the repository then download and extract necessary resource files:
```bash
git clone https://github.com/FunctionLab/Seal.git
cd Seal
sh ./download_resources.sh
```

## Usage

### Context-specific prediction of variant effects on expression

Command line ([example bash script](test/predict/test_var_predict.sh)):
```bash
python seal_predict.py --vcf_file <variant vcf file> --model_info_file <Seal model summary file> --out_file <Output model prediction file>
```

Arguments:
- `--vcf_file`: input VCF file (hg19-based coordinate; [example](test/predict/test_var.vcf))
- `--model_info_file`: input Seal model info file (contains pre-trained and fine-tuned model file location and hidden layer info; [example](model/tissue_state_early_fetal_to_adult/tissue_state_early_fetal_to_adult_seal_model_summary.txt)) 
- `--out_file`: output result file of variant effect predictions ([example](test/predict/test_var_tissue_state_effect_pred.tsv)) 

Notes:
- We provided three trained and evaluated Seal models predicting gene expression of brain development at both tissue and cell type resolution. The first model can predict variant effects on gene expression under 122 tissue states of 7 developmental stages from early fetal to adulthood ([model info file](model/tissue_state_early_fetal_to_adult/tissue_state_early_fetal_to_adult_seal_model_summary.txt)). The second model can predict variant effects on gene expression under 598 cell states of early fetal stage ([model info file](model/cell_state_early_fetal/cell_state_early_fetal_seal_model_summary.txt)). The third model can predict variant effects on gene expression under 82 cell states of 6 developmental stages from mid fetal to adulthood ([model info file](model/cell_state_mid_fetal_to_adult/cell_state_mid_fetal_to_adult_seal_model_summary.txt)). For detailed information about the cell states, please check [the annotation file](resource/cell_state_annotation.xlsx).
- Our models were trained with sequence from the hg19 reference genome assembly. Users can use [UCSC lift genome annotations](https://genome.ucsc.edu/cgi-bin/hgLiftOver) for liftover coordinates of other assembly to hg19. Alternatively, users can also replace the input gene annotation BED file (`--gene_bed_file` argument) and input reference genome fasta file (`--ref_genome_file` argument) with files of the preferred assembly.  

### Training a sequence-to-expression transfer learning model from scratch 

Command line ([example bash script](test/train/test_train_from_scratch.sh)):
```bash
python seal_train.py --general_exp_file <general context expression file> --n_latent <number of hidden neurons> --lr_pretrain <pre-training learning rate> --l2_pretrain <pre-training L2 regularization factor> --general_context_group <general context group info file> --spec_gene_weight <weight assigned to specific genes> --finetune_exp_file <specific context expression file> --lr_finetune <fine-tuning learning rate> --l2_finetune <fine-tuning L2 regularization factor> --specific_context_group <specific context group info file> --out_name <output file location>
```

Arguments:
- `--general_exp_file`: expression matrix .tsv file of general contexts for pre-training. First column contains gene id. Second to last columns contain normalized expression value. ([example](resource/geneanno.exp.csv_general_brain.tsv))
- `--n_latent`: number of hidden neurons for the Module 3 transfer learning neural network model of Seal framework 
- `--lr_pretrain`: learning rate in pre-training for the Module 3 transfer learning neural network model of Seal framework
- `--l2_pretrain`: L2 regularization factor in pre-training for the Module 3 transfer learning neural network model of Seal framework
- `--general_context_group`: group info .tsv file of general contexts for gene-weighting of neural network loss function. First column contains the group name. Second column contains the column ID among expression matrix columns. If provided, gene weights will be assigned separately for each context group, based on the expression variation within each group. If not provided (default setting), genes weights will be assigned based on the expression variation across all contexts ([example](resource/brain_dev_exp_tissue_group.tsv)). 
- `--spec_gene_weight`: the weight score assigned to genes with high expression variation (by default, score of `1/spec_gene_weight` will be assigned to genes with low expression variation). 
- `--finetune_exp_file`: expression matrix .tsv file of general contexts for fine-tuning. Same format as `--general_exp_file`
- `--lr_finetune`: learning rate in fine-tuning for the Module 3 transfer learning neural network model of Seal framework
- `--l2_finetune`: L2 regularization factor in fine-tuning for the Module 3 transfer learning neural network model of Seal framework
- `--specific_context_group`:  group info .tsv file of specific contexts for gene-weighting of neural network loss function. Same format as `--general_context_group`
- `--out_name`: path for output. All output files will be named and stored based on the specific path. 

### Fine-tuning a pre-trained model for predicting gene expression under specific contexts

Command line ([example bash script](test/train/test_train_from_pretrained.sh)): 
```bash
python seal_train.py --general_exp_file <general context expression file> --pretrained <True> --pretrained_name <pre-trained file location> --n_latent <number of hidden neurons> --spec_gene_weight <weight assigned to specific genes> --finetune_exp_file <specific context expression file> --lr_finetune <fine-tuning learning rate> --l2_finetune <fine-tuning L2 regularization factor> --specific_context_group <specific context group info file> --out_name <output file location>
```

Additional arguments: 
- `--pretrained`: bool specifying whether the pretrained model exists (True in this case)
- `--pretrained_name`: path where pre-trained files are stored. Same format as `--out_name`. Pre-trained files will be loaded based on the specified path and our naming scheme.

## Help
Please post in the Github issues or e-mail Yun Hao [yhao@flatironinstitute.org](mailto:yhao@flatironinstitute.org) with any questions about the repository, requests for more data, etc.

