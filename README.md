# Cumulative Link Models

<p align=center>
    <img src=res/clm_thumbnail.png height=400>
</p>

Code for cumulative link models for ordinal regression that support differentiable learning ala TensorFlow and sklearn.

<blockquote>
<p>
<i>Semi-supervised Ordinal Regression via Cumulative Link Models for Predicting In-Hospital Length-of-Stay</i>.
 <br />
Alexander A. Lobo, Preetish Rath, Micheal C. Hughes
 <br />
<!-- (35):1019−1041, 2005. -->
 <br />
<!-- PDF available: <a href="https://www.jmlr.org/papers/volume6/chu05a/chu05a.pdf">https://www.jmlr.org/papers/volume6/chu05a/chu05a.pdf</a> -->
</p>
</blockquote>

To appear at [IMLH 2023](https://sites.google.com/view/imlh2023/home?authuser=1).

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
of the diabetes.

The [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) from sklearn is used.

The raw target attribute is a reak valuequantitative measure of the disease 
progression one year after baseline. A `KBinsDiscretizer` is used to bin the
continuous target data and encode ordinal labels such that each bin contains
the same number of data points.

To run the notebook demo:

1. Navigate to the notebook located at
```sh
cumulative-link-models/demo/ordinal_regression_via_CLMs.ipynb
```
2. Set the kernel as the conda environment (refer above for instructions)
3. Run the notebook to create, train, and assess the model

## Experiments

## Citing
To cite this repository, please cite the published paper:
```
@inproceedings{Lobo_Semi-supervised_Ordinal_Regression_2023,
  author = {Lobo, Alexander A. and Rath, Preetish and Hughes, Michael C.},
  booktitle = {3rd Workshop on Interpretable Machine Learning in Healthcare (IMLH) at International Conference on Machine Learning (ICML)},
  title = {{Semi-supervised Ordinal Regression via Cumulative Link Models for Predicting In-Hospital Length-of-Stay}},
  year = {2023}
}
```