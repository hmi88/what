# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

Pytorch implementation of ["What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017](https://arxiv.org/abs/1703.04977) 

[[Image]]

[[paper 3line summary]]

[[Notice]]



## 1. Usage

```
# Data Tree
config.data_dir/
└── config.data_name/

# Project Tree
WHAT
├── WHAT_src/
│       ├── data/ *.py
│       ├── loss/ *.py
│       ├── model/ *.py
│       └── *.py
└── WHAT_exp/
         ├── log/
         ├── model/
         └── save/         
```



### 1.1  Train

```
# L2 loss only 
python train.py --uncertainty "normal"

# Epistemic / Aleatoric 
python train.py --uncertainty ["epistemic", "aleatoric"]

# Epistemic + Aleatoric
python train.py --uncertainty "combined"
```



### 1.2 Test

```
# L2 loss only 
python train.py --is_train false --uncertainty "normal"

# Epistemic
python train.py --is_train false --uncertainty "epistemic" --n_samples 25 [or 5, 50]

# Aleatoric
python train.py --is_train false --uncertainty "aleatoric" 

# Epistemic + Aleatoric
python train.py --is_train false --uncertainty "combined" --n_samples 25 [or 5, 50]
```



### 1.3 Requirements

- Pytorch >= 1.0
- Torchvision
- distutils



## 2. Experiment

### 2.1 Dataset

fashion-mnist



### 2.2 Loss



### 2.2 Results

