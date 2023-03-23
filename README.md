# TSVLC
Repository for the paper: Teaching Structured Vision &amp; Language Concepts to Vision &amp; Language Models

link: https://arxiv.org/abs/2211.11733

model checkpoint for [model,RB] negatives: /dccstor/sivandov1/dev/open_clip_vl/Outputs/cor_cc3m_both_negs

# Installation:
## Requirements
1. Linux machine
1. At least one NVIDIA GPU
1. At least CUDA 10.2
1. Anaconda (Installation instructions: https://docs.anaconda.com/anaconda/install/)
## Install Dependencies
Clone the repository:
`git clone TSVLC`
Enter the directory:
`cd TSVLC`
Create and activate the conda environment:
```shell script
conda deactivate # deactivate any active environments
conda create -n vl python=3.6 # install the conda environment with conda dependencies
conda activate vl # activate the environment
conda install -c conda-forge libjpeg-turbo
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt # install pip requirements
```

# Data Preperations
Download Conceptual Captions 3M training and validation splits from https://ai.google.com/research/ConceptualCaptions/download
After data preperation, place the data in `<TSVLC_ROOT_DIR>/cc3m_data/training` and `<TSVLC_ROOT_DIR>/cc3m_data/validation`

# Training and Evaluation

## Run the training script
To train a network with the RB negative generation - run the following command from the project root directory:
```shell script
cd src
python3 training/main.py --name exp_name --vl_negs --lora 4 --neg_type word_replacement --pretrained openai
```

To train a network with the RB + Bert based negatives generation - run the following command from the project root directory:
```shell script
cd src
python3 training/main.py --name exp_name --vl_negs --lora 4 --lr 5e-06 --neg_type rand_both --auto_neg_types NOUN ADP ADJ VERB --pretrained openai
```

## Run the evaluation script
To prepare the vl checklist evaluate results for the experiment `exp_name` run the following command:
```shell script
python3 training/main.py  --lora 4 --pretrained openai --eval_recall --eval_vl_cklist --eval_only --resume ./Outputs/exp_name_checkpoint.pt
```
## print vl checklist results
To calculate the vl checklist results for the experiment `exp_name` run the following command:
```shell script
python3 training/main.py  --lora 4 --pretrained openai --eval_recall --eval_vl_cklist --eval_only --resume ./Outputs/exp_name_checkpoint.pt
```

