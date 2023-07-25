# Cumulative Link Models

<p align=center>
    <img src=res/clm_thumbnail.png width=100%>
    <figcaption><b><i>Fig. 1:</i></b> Cumulative link model likelihood depictions with different link functions and scale parameters.</figcaption>
</p>

Code for cumulative link models for ordinal regression that support differentiable learning ala TensorFlow and sklearn outlined in the following paper:

<blockquote>
<p>
<i>Semi-supervised Ordinal Regression via Cumulative Link Models for Predicting In-Hospital Length-of-Stay</i>.
 <br />
Alexander A. Lobo, Preetish Rath, Micheal C. Hughes
 <br />
3rd Workshop on Interpretable Machine Learning in Healthcare (IMLH) at International Conference on Machine Learning (ICML)
 <br />
2023
 <br />
PDF available: <a href="https://openreview.net/forum?id=pDDKtCklZy">https://openreview.net/forum?id=pDDKtCklZy</a>
</p>
</blockquote>

Appeared at [IMLH 2023](https://sites.google.com/view/imlh2023/home?authuser=1).

## Contents
1. [Setup](#setup)
2. [Demo](#demo)
3. [Experiments](#experiments)
4. [Citing](#citing)

## Setup

### Install Anaconda
Follow the instructions here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### Create environment
```sh
conda env create -f environment.yml
```

## Demo

### Diabetes Disease Progression Prediction

This demo creates a simple network model in TensorFlow to predict the progression
of diabetes amongst patients.

<p align=center>
    <img src=res/tensorflow_clm_model.png width=75%>
    <figcaption><b><i>Fig. 2:</i></b> Simple CLM network model representation used to predict diabetes ordinal progression category.</figcaption>
</p>

The [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) from sklearn is used.

The raw target attribute is a real value quantitative measure of the disease 
progression one year after baseline. A `KBinsDiscretizer` is used to bin the
continuous target data and encode ordinal labels such that each bin contains
the same number of data points.

To run the notebook demo:

1. Navigate to the notebook located at
```sh
demo/ordinal_regression_via_CLMs.ipynb
```
2. Set the kernel as the conda environment (refer above for setup instructions)
3. Run the notebook to create, train, and assess the model

### Performance Improvement with Cutpoints Learning

This demo shows the comparison of cumulative link model performance with and
without learning the cutpoints on 2-dimensional toydata.

To run the notebook demo:

1. Navigate to the notebook located at
```sh
demo/CLM_cutpoint_training.ipynb
```
2. Set the kernel as the conda environment (refer above for setup instructions)
3. Run the notebook to create, train, and assess the models

## Experiments

## Citing
To cite this repository, please cite the published paper:
```
@inproceedings{lobo2023semisupervised,
title={Semi-supervised Ordinal Regression via Cumulative Link Models for Predicting In-Hospital Length-of-Stay},
author={Alexander Arjun Lobo and Preetish Rath and Michael C Hughes},
booktitle={ICML 3rd Workshop on Interpretable Machine Learning in Healthcare (IMLH) },
year={2023},
url={https://openreview.net/forum?id=pDDKtCklZy}
}
```