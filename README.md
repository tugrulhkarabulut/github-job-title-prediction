# Job Title Prediction of Github Users

This repository contains implementations for a machine learning pipeilne to predict job titles of github users. Both classical ML models and graph deep learning models are available. [Github Social Network](https://snap.stanford.edu/data/github-social.html) is used as the reference dataset for the user subset but a new set of features related to user's repositories, company and descriptive statistics are extracted.

# Setup

You need to download the Github Social Network. To be able to make requests to Github Rest API, you need to install [Github CLI](https://cli.github.com/). 

## Libraries Used

- scikit-learn
- Pandas
- Pytorch
- DGL

# How to Run

Get authenticated in Github with:

```bash
gh auth login
```

## Getting the Data

Following functions from 'utils.py' should be called to get the necessary data from Github:

```python

get_absent_users_from_api
get_user_relations_from_api
get_user_repos_from_api

```

## Labeling & Feature Extraction

Execute 'edge analysis.ipynb', 'label_analysis.ipynb' and 'feature_extraction.ipynb' notebooks.

## Run Model

You can use the "run.py" function to train and evaluate models.

Example usage:

```bash
python run.py --model GraphSAGE --feature-selection select_from_model --select-from extra_trees --undirected --h-feats 400
```

All options:


```bash
options:
  -h, --help            show this help message and exit
  --model {NaiveBayes,LogisticRegression,GCN,GraphSAGE}
                        Model name.
  --lr-max-iter LR_MAX_ITER
                        Logistic Regression iteration.
  --lr LR               Learning rate for GCNs.
  --h-feats H_FEATS     Hidden units.
  --epochs EPOCHS       Number of epochs.
  --patience PATIENCE   Number of iterations to wait for improvement before early stopping.
  --undirected          Make the graph undirected.
  --feature-selection {None,variance,select_from_model}
                        Feature selection method.
  --variance-threshold VARIANCE_THRESHOLD
                        Threshold value for variance feature selection.
  --select-from {svc,extra_trees}
                        Select features according to given model.
  --n-splits {1,5}      Number of splits for k-fold cross-validation.
  --neighborhood-features {mean,max}
                        Neighborhood aggregation function for non-graph models.
```


## Experiments

### Results

| Model                   | Weighted F-1              |
|-------------------------|---------------------------|
| \#1 Logistic Regression | 0.752 ± 0.009          |
| \#2 Naive Bayes         | 0.736 ± 0.007          |
| \#3 GraphSAGE           | **0.762 ± 0.008** |
| \#4 GCN                 | 0.758 ± 0.006          |

### Run Experiments
Run experiments.sh to reproduce the results of the experiments in this study.