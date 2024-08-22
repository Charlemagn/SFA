# Modeling Selective Feature Attention for Lightweight Text Matching

This repository is the implementation presented in the paper 
["Modeling Selective Feature Attention for Lightweight Text Matching"](https://www.ijcai.org/proceedings/2024/0732.pdf)

## How to
### Install the package
To use the model defined in this repository, you will first need to install PyTorch on your machine by following the steps
described on the package's [official page](https://pytorch.org/get-started/locally/) (this step is only necessary if you use
Windows).
Then, to install the dependencies necessary to run the model, simply execute the command `pip install --upgrade .` from within
the cloned repository (at the root, and preferably inside of a [virtual environment](https://docs.python.org/3/tutorial/venv.html)).

### Fetch the data to train and test the model
The *fetch_data.py* script located in the *scripts/* folder of this repository can be used to download some NLI dataset and
pretrained word embeddings. By default, the script fetches the [SNLI](https://nlp.stanford.edu/projects/snli/) corpus and
the [GloVe 840B 300d](https://nlp.stanford.edu/projects/glove/) embeddings. Other datasets can be downloaded by simply passing
their URL as argument to the script (for example, the [MultNLI dataset](https://www.nyu.edu/projects/bowman/multinli/)).

The script's usage is the following:
```
load_data.py [-h] [--dataset_url DATASET_URL]
              [--embeddings_url EMBEDDINGS_URL]
              [--target_dir TARGET_DIR]
```
where `target_dir` is the path to a directory where the downloaded data must be saved (defaults to *../data/*).

For MultiNLI, the matched and mismatched test sets need to be manually downloaded from Kaggle and the corresponding .txt files 
copied in the *multinli_1.0* dataset folder.

### Preprocess the data
Before the downloaded corpus and embeddings can be used in the ESIM model, they need to be preprocessed. This can be done with
the *preprocess_\*.py* scripts in the *scripts/preprocessing* folder of this repository. The *preprocess_snli.py* script can be 
used to preprocess SNLI, *preprocess_mnli.py* to preprocess MultiNLI, and *preprocess_bnli.py* to preprocess the Breaking NLI 
(BNLI) dataset. Note that when calling the script fot BNLI, the SNLI data should have been preprocessed first, so that the 
worddict produced for it can be used on BNLI.

The scripts' usage is the following (replace the \* with *snli*, *mnli* or *bnli*):
```
preprocess_*.py [-h] [--config CONFIG]
```
where `config` is the path to a configuration file defining the parameters to be used for preprocessing. Default 
configuration files can be found in the *config/preprocessing* folder of this repository.

### Train the model
The *train_\*.py* scripts in the *scripts/training* folder can be used to train the ESIM model on some training data and 
validate it on some validation data.

The script's usage is the following (replace the \* with *snli* or *mnli*):
```
train_*.py [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
```
where `config` is a configuration file (default ones are located in the *config/training* folder), and `checkpoint` is an 
optional checkpoint from which training can be resumed. Checkpoints are created by the script after each training epoch, with 
the name *esim_\*.pth.tar*, where '\*' indicates the epoch's number.

### Test the model
The *test_\*.py* scripts in the *scripts/testing* folder can be used to test a pretrained ESIM model on some test data.

To test on SNLI, use the *test_snli.py* script as follows:
```
test_snli.py [-h] test_data checkpoint
```
where `test_data` is the path to some preprocessed test set, and `checkpoint` is the path to a checkpoint produced by the 
*train_snli.py* script (either one of the checkpoints created after the training epochs, or the best model seen during 
training, which is saved in *data/checkpoints/SNLI/best.pth.tar* - the difference between the *esim_\*.pth.tar* files and 
*best.pth.tar* is that the latter cannot be used to resume training, as it doesn't contain the optimizer's state).

The *test_snli.py* script can also be used on the Breaking NLI dataset with a model pretrained on SNLI.

To test on MultiNLI, use the *test_mnli.py* script as follows:
```
test_mnli.py [-h] [--config CONFIG] checkpoint
```
where `config` is a configuration file (a default one is available in *config/testing*) and `checkpoint` is a checkpoint 
produced by the *train_mnli.py* script.

The *test_mnli.py* script makes predictions on MultiNLI's matched and mismatched test sets and saves them in .csv files.
To get the classification accuracy associated to the model's predictions, the .csv files it produces need to be submitted
to the Kaggle competitions for MultiNLI.

### Citation

If you find the code is helpful, please cite:

```
@article{zangmodeling,
  title={Modeling Selective Feature Attention for Lightweight Text Matching},
  author={Zang, Jianxiang and Liu, Hui}
}
```
