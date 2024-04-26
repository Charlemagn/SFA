# Modeling Selective Feature Attention for Representation-based Siamese Text Matching
This repository is the code implementation of the paper:
["Modeling Selective Feature Attention for Representation-based Siamese Text Matching"](https://arxiv.org/pdf/2404.16776) 

Our proposed [FA and SFA](https://github.com/hggzjx/SFA/blob/main/base_model/our_layers.py) blocks are plug-and-play modules.

## Setup Instructions
### Create a conda environment:
```
conda create -n SFA4Text_matching python=3.9
conda activate SFA4Text_matching
```
### Install dependencies:
```
pip install -r requirements.txt
```
### Load the data and preprocess the data
The *load_data.py* script located in the *scr/* folder of this repository can be used to download some text matching dataset 
and pretrained word embeddings. By default, the script loads the most classic dataset [SNLI](https://nlp.stanford.edu/projects/snli/) 
corpus and the [GloVe 840B 300d](https://nlp.stanford.edu/projects/glove/) embeddings. where `target_dir` is the path to a directory 
where the downloaded data must be saved (defaults to *../data/*). Before using the downloaded corpus and embeddings for the base model, 
they need to be preprocessed using the *preprocess.py* script in the src/preprocess folder of this repository. where `config` 
is the path to a configuration file defining the parameters to be used for preprocessing. Default configuration files can be found in the *config/preprocess* folder 
of this repository.
```
cd src
load_data.py [--dataset_url DATASET_URL][--embeddings_url EMBEDDINGS_URL][--target_dir TARGET_DIR]
preprocess.py [--config CONFIG]
```
### Train and Test the model with FA or SFA
```
train.py [--config CONFIG] [--checkpoint CHECKPOINT]
test.py [--test_data Test_DATA] [--checkpoint CHECKPOINT]
```
where `config` is a configuration file (default ones are located in the *config/train* folder), and `checkpoint` is an 
optional checkpoint from which training can be resumed. Checkpoints are created by the script after each training epoch.
This repository provides a demonstration of applying FA or SFA to matching models. The choice of whether to apply FA or 
SFA can be made in the *config/train* directory.


