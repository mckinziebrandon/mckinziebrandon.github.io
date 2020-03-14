---
layout: post
title:  "TODO"
date:   2019-08-24
excerpt: "Historical look at usage of transformers for language modeling"
tags: [transformer, language model]
comments: true

---

# Progression of Transformers for Language Modeling

Take a look at how things have evolved along the path of

1. Vaswani 2017 Attention is all you Need
2. Radford 2018 GPT1
3. Radfor 2019 GPT2
4. Dai 2019 Transformer XL

## Contributions by Paper

* GPT-1
  * Gaussian Error Linear Unit for FFN activation function (instead of ReLU). 
  * Learned positional embeddings instead of sinusoidal.
  * Cosine learning rate schedule.
  * Custom Adam implementation.
* GPT-2
  * Layer norm on _inputs_ of each sub-block instead of on the residual output. i.e. instead of `LayerNorm(x + Sublayer(x))`, do `x + Sublayer(LayerNorm(x))`. 
  * Adds another layer normalization after the final self-attention block. 
  * Scale weights of residual layers at initialization by a factor of $1/\sqrt{N}$ where $N$ is the number of residual layers (note that I cannot actually find this anywhere in their code...).  
* Transformer-XL
  * Segment-level recurrence
  * Relative positional encoding
  * Adaptive input representations (Baevski 2018)
  * Adaptive softmax (Grave 2016)

--------------------------------------------------------

## Miscellaneous

- Dividing dot products in the attention mechanism by $$\sqrt{d_k}$$ 
  - Assuming $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$ for all elements of $\mathbf{q}$ and $\mathbf{k}$, and variance of 1 similarly, we see that $\mathbf q \cdot \mathbf k$ has a standard deviation of $\sqrt{d_k}$. 
- Why is Bahdanau attention called "additive" attention?