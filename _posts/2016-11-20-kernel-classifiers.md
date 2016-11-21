---
layout: post
title:  "Polynomial and Gaussian Kernels"
date:   2016-11-07
excerpt: "A mini-project that implements a simple classifier with polynomial and Gaussian kernels."
tags: [markdown,  jekyll, test]
comments: false
---

# Problem Overview

Two-class classification. 
* __Class 1 (y = 1)__: 
    \\[ (x_1, x_2)  = (8\cos\theta + w_1, 8\sin\theta + w_2) \\]
     \\[   \theta \sim Unif(0, 2\pi) \\]
       \\[ w_1, w_2 \sim \mathcal{N}(0, 1) \\]

* __Class 2 (y = -1)__: 
    \\[ (x_1, x_2) = (v_1, v_2)  \\]
 \\[   v_1, v_2 \sim \mathcal{N}(0, 1)  \\]


```python
import numpy as np
import matplotlib.pyplot as plt

# Construct class 1:(x_1, x_2)=(8\cos\theta + w_1, 8\sin\theta + w_2) 
w = np.random.normal(0, 1, (100, 2))
theta = np.random.uniform(2 * np.pi, size=(100,))
x_class1 = 8.*np.hstack((np.cos(theta)[:, None], 
                         np.sin(theta)[:, None]))
x_class1 += w

# Construct class 2: (x_1, x_2) = (v_1, v_2)
v = np.random.normal(0, 1, (100, 2))
x_class2 = v

# Combine to obtain data matrix X, and also make label vector y.shape(200, 1). 
X = np.vstack((x_class1, x_class2))
y = np.ones((200, 1))
y[100:] = -1
```


```python
# _________________ Plotting _________________
plt.style.use('ggplot')
fig, ax = plt.subplots(1, 1)

plt.scatter(X[:100, 0], X[:100, 1], c='r', label='y = +1')
plt.scatter(X[100:, 0], X[100:, 1], c='b', label='y = -1')

ax.legend(loc='best', frameon=False)
fig.suptitle('100 Samples of Each Class', 
             fontsize=14, fontweight='bold')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.show()
```


![png]({{site.url}}/assets/img/output_2_0.png)



```python
import pdb
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, alpha, 
                          kernel='poly', 
                          gamma=10., 
                          resolution=1.):
    """ Disclaimer: This function is an adapted version 
    from several in the textbook "Python Machine Learning". 
    """
    # Setup marker generator and color map. 
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Get the min/max (of each) feature values in the dataset. 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                          np.arange(x2_min, x2_max, resolution)) 
    
    # Assign class prediction on grid of (x1, x2) values.
    X_test = np.array([xx1.ravel(), xx2.ravel()]).T
    
    if kernel == 'poly':
        Z = K(X_test, X, kernel) @ alpha
    else:
        Z = np.zeros((X.shape[0], X_test.shape[0]))
        for i in np.arange(X.shape[0]):
            for j in np.arange(X_test.shape[0]):
                dists = (X[i] - X_test[j]).T @ (X[i] - X_test[j])
                Z[i, j] = np.exp(- gamma * dists)
        Z = Z.T
        Z= Z @ alpha
    
    Z = np.where(Z >= 0, 1, -1)
    Z = Z.reshape(xx1.shape)
    

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # Plot class samples. 
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y.flatten() == cl, 0], 
                    y=X[y.flatten() == cl, 1], 
                    alpha=0.8, c=cmap(idx), 
                    marker=markers[idx], label=cl)
        

def K(X, Z, kernel='poly', gamma=10.):
    from scipy.spatial.distance import pdist, squareform
    I = np.ones((X.shape[0], Z.shape[0]))
    if kernel == 'poly':
        return (I + X @ Z.T)**2
    elif kernel == 'gauss':
        distances = squareform(pdist(X, 'sqeuclidean'))
        return np.exp(-gamma * distances)
    
def alpha_star(X, y, kernel='poly', reg=1e-6):
    return np.linalg.solve((K(X, X, kernel) + reg * np.eye(X.shape[0])), y )

def get_accuracy(Y_pred, Y_true):
    return np.sum(Y_true == Y_pred)  /  Y_true.shape[0]

def predict(X):
    return np.where((K(X) @ alpha(K(X), y)) > 0, 1, -1)
```

## Polynomial Kernel

We can write the kernel matrix in the form \\( K = \Phi(X)\Phi(X)^T \\), where 
\\[ \Phi(x) = [1, ~ \sqrt{2}x_1, ~ \sqrt{2}x_2, ~ \sqrt{2}x_1x_2, ~ x_1^2, ~ x_2^2]^T \\]
Our task is to find \\( \alpha \\) that minimizes our objective function: \\[ ||K\alpha - y||^2 + \lambda \alpha^T K \alpha \\]

Setting the gradient to zero and solving for \\( \alpha \\) yields
\\[ \alpha = \left(K + \lambda I \right)^{-1} y \\]
\\[ K = (1 + XX^T)^2 \\]

Below, we plot the training data and the corresponding decision boundary. Our training accuracy is 99.5%, given \\( \lambda = 1e^{-6} \\). 


```python
reg = 1e-6
alpha = alpha_star(X, y, kernel='poly')
pred = np.where(K(X, X) @ alpha > 0, 1, -1)
print("Training accuracy is", 100. * get_accuracy(pred, y.astype(int)), "percent.")

# 2.
plot_decision_regions(X, y, alpha, kernel='poly')
plt.legend(loc='upper left')
plt.show()
```

    Training accuracy is 99.5 percent.



![png]({{site.url}}/assets/img/output_5_1.png)


## Gaussian Kernel

We repeat this process but now with the Gaussian kernel,
\\[
k(x, z) = \exp\big(-\gamma || x - z||^2 \big)
\\]
and with \\( \lambda = 1e^{-4} \\). Below are the plots corresponding to \\( \gamma = 10, 0.1, 0.001,\\) and in that order. The corresponding training accuracies for each \\( \gamma \\) is 100%, 100%, and 99.5%. 


```python
reg = 1e-4
for g in [10, 0.1, 0.001]:
    K_train = K(X, X, kernel='gauss', gamma=g)
    alpha = np.linalg.solve((K_train + reg * np.eye(X.shape[0])), y)
    pred = np.where(K_train @ alpha > 0, 1, -1)
    print("Training accuracy is", 
          100. * get_accuracy(pred, y.astype(int)), "percent.")
    plot_decision_regions(X, y, alpha, kernel='gauss', gamma=g)
    plt.legend(loc='upper left')
    plt.show()
```

    Training accuracy is 100.0 percent.



![png]({{site.url}}/assets/img/output_7_1.png)


    Training accuracy is 100.0 percent.



![png]({{site.url}}/assets/img/output_7_3.png)


    Training accuracy is 100.0 percent.



![png]({{site.url}}/assets/img/output_7_5.png)

