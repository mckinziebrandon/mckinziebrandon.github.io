---
layout: post
title:  "ML Foundations: Models"
date:   2020-02-09
excerpt: "Overview of the must-know ML models"
tags: [machine learning]
comments: true
---

# Supervised Learning 

## Linear Regression

Given feature vector \\(\mathbf{x} \in \mathbb{R}^d\\) and parameters \\(\mathbf{\beta} \in \mathbb{R}^d\\), predicts scalar \\(y \in \mathbb{R}\\) via:

\\[y = \mathbf{\beta}^T\mathbf{x}\\]

## Logistic Regression

Essentially an extension of linear regression for classification (discrete labels). Let \\(K\\) denote the number of classes.

\\[\begin{align}
    y &= arg\max_k Pr(k \mid x)  
    \\\\ 
    Pr(k \mid x) &= \frac{\exp{\mathbf{\beta}_k^T\mathbf{x}}}{1+\sum\_{\ell=1}^{K-1} \exp{\mathbf{\beta}\_{\ell}^T\mathbf{x}}}  \quad (1 \le k \le K -1)  
    \\\\ 
    Pr(K \mid x) &= \frac{1}{1+\sum\_{\ell=1}^{K-1} \exp{\mathbf{\beta}\_{\ell}^T\mathbf{x}}} 
\end{align}\\]

## Support Vector Machine (SVM)

Extension of the _support vector classifier_ (SVC) in order to accomodate non-linear class boundaries. The SVC is itself an extension of the _maximum margin classifier_ that drops the constraint that all classes are linearly separable. SVMs are intended for binary classification, but they can be extended for the multi-class case.

In \\(\mathbb{R}^n\\), a __hyperplane__ is a flat affine subspace of dimension \\(n-1\\). It is defined as the set of points \\(\mathbf{x} \in \mathbb{R}^n\\) for which

\\[ \beta_0 + \mathbf{\beta}^T \mathbf{x} = 0\\]

Without loss of generality, we will be enforcing the constraint that \\(\|\|\mathbf{\beta}\|\|_2 = 1\\). Note that \\(\|\beta_0\|\\) is the absolute distance from the origin to the hyperplane, along the direction \\(\beta\\). When \\(\beta_0 = 0\\) (i.e. the hyperplane goes through the origin), this is all vectors \\(\mathbf{x}\\) that are orthogonal to \\(\mathbf{\beta}\\). More generally, regardless of the value of \\(\beta_0\\), the vector \\(\mathbf{\beta}\\) is orthogonal to the hyperplane. Note that this is NOT the same as saying "\\(\mathbf{\beta}\\) is orthogonal to all vectors \\(\mathbf{x}\\) on the hyperplane".

A key property of a hyperplane is that it divides the space in two. Suppose, instead of having binary labels 0 and 1 like we did in logistic regression, our labels are -1 and +1. Also suppose that all of our labeled data points \\(\mathbf{x}\\) can be perfectly separated by a hyperplane, such that all points with label -1 are on one side and all points with label +1 are on the other side. We can state this formally as, for any labeled example \\((\mathbf{x}_i, y_i)\\)
\\[y_i(\beta_0 + \mathbf{\beta}^T\mathbf{x}_i) > 0\\]

In general, if our data is linearly separable, many such hyperplanes will exist. The _maximum margin classifier_ tries to find the separating hyperplane that is farthest from the training observations. The *margin* is defined as the smallest distance from the hyperplane to a training data point. For any input \\(\mathbf{x}\\), the MMC predicts \\(y=sign(\beta_0 + \mathbf{\beta}^T\mathbf{x})\\)

What if our data isn't linearly separable but we still want to classify points using the notion of a maximum margin (in some sense)? Clearly, we will need to relax the constraint that every single training point must be on one side of the hyperplane corresponding to its class label. In other words, we'll have to use a "soft margin". This describes the __Support Vector Classifier__ (SVC). It classifies using inner products over the set of _support vectors_ \\(\mathcal{S}\\):
\\[f(x)=\beta_0 + \sum\_{i \in \mathcal S} \alpha_i \langle x, x_i\rangle\\]

Finally, we can further generalize the SVC to handle data that is non-linearly separable. As is often done, this is accomplished with _kernels_. The SVC is just a special case where the linear kernel is used. SVMs generalize this to any (possibly non-linear) kernel K:
\\[f(x)=\beta_0 + \sum\_{i \in \mathcal S} \alpha_i K(x, x_i)\\]

* KNN
* Linear Discriminant Analysis (LDA) (generative)
* Quadratic Discriminant Analysis (QDA) (generative)
* Neural Networks
* Decision Trees / Random Forests / boosting / etc
* Perceptron 
* Naive Bayes

# Unsupervised Learning

* K-Means
* PCA
* SVD / matrix factorization

Less must-know but still important:
* Hierarchical Clustering
* Spectral Clustering

# Reinforcement Learning

* MDPs
* Games

# Generative Models

* Autoregressive
    * Masked Autoencoder for Distribution Estimation (MADE)
* Latent Variable Models
    * Mixture of Gaussians (MoG)
    * Variational Autoencoder (VAE)
* Normalizing Flow Models
    * Nonlinear Independent Components Estimation (NICE)
    * Real-NVP
    * Masked Autoregressive Flow (MAF)
    * Inverse Autoregressive Flow (IAF)
* GANs
* Energy-Based Models
* Linear Discriminant Analysis
* Latent Dirichlet Allocation

# Graphical Models

* Bayesian Networks
* Undirected Graphical Models
* Hidden Markov Models

# Feature Stuff?
where should this even go

* Kernel stuff
* Splines

# Other (TODO)

* CSPs

# Review of Where Stuff Overlaps

TODO
