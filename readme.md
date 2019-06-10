# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

Pytorch implementation of ["What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017](https://arxiv.org/abs/1703.04977) 

[[Image]]

[[paper 3line summary]]

[[Notice]]



## 0. Requirments





## 1. Usage

### 1.1  Train

```

```



### 1.2 Test

### 1.3 Requirements

- Pytorch
- 



## 2. Implmentation detail

### 2.1 Dataset

### 2.2 Architecture

### 2.3 Loss

**2.3.1 Epistemic Uncertainty**

```python
def epistemic_loss():
  return
```



**2.3.2 Heteroscedastic Aleatoric Uncertainty**

```python
def aleatoric_loss():
    return	
```



**2.3.3 Combining Aleatoric and Epistemic Uncertainty***

```python
def combine_loss():
	return
```



## 3. Experiment

### 3.1 Results



### 3.2 uncertainty effect

|      | L2 only | L2 + Epistemic | L2 + Aleatoric | L2 + Epistemic |
| ---- | ------- | -------------- | -------------- | -------------- |
| rmse |         |                |                |                |
| pil  |         |                |                |                |
| Cicl |         |                |                |                |



### 3.3 data crosscheck

| Train     | Test    | rms  | ep_var | al-var |
| --------- | ------- | ---- | ------ | ------ |
| MNIST / 4 | MNIST   |      |        |        |
| MNIST     | MNIST   |      |        |        |
| MNIST     | Fashion |      |        |        |



## 4. Reference



## 5. Author

Kiheum Cho 

