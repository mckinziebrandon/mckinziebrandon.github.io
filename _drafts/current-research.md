---
layout: post
title:  "Automatic Architecture Generation for Deep Neural Networks"
date:   2016-11-27
excerpt: "A brief overview of my current research."
tags: [research,  deep learning, python, tensorflow]
comments: false
---

# Motivation

One of the challenges in building an effective deep neural network is, of course, figuring out how to build it! There are an
infinite number of possible combinations for hyperparameters such as . . . 

* number of layers (the "depth")
* number of parallel layers (the "width")
* activation functions for each (computational) layer
* learning rate(s) and/or momentum 

. . . and the list goes on and on. So how do the experts do it? You may be surprised to find out that machine learning is still a lot of trial-and-error. However, this just doesn't feel right. Surely there is a way we can automate this process. 


# Project Overview

My current research, led by Professor Dawn Song's research group at UC Berkeley, aims to solve this problem. Rather than having humans tinker around with designing deep neural networks, this project aims to write software that can learn to build deep networks on its own.

# Some of My Work

Here I'll log some of the more interesting problems, in my opinion, that I've encountered in this project. 

## Early Stopping

Say you're a sentient machine trying to figure out the best network build. As we've discussed, you have about an infinite number of ways to approach this. Regardless of what you choose, you'll certainly need a way to *evaluate* how well you've done after constructing a given architecture.
Unfortunately, even for a fast AI such as yourself, training and evaluation can take a *really* long time. 

A common technique for speeding up the training and evaluation, known as **early stopping**, involving periodically "peeking under the hood" of the network as it learns, and determining whether or not (a) it is mostly done learning (negligible improvement), or (b) it has "learned enough," where it
is up to the designer to determine what "enough" means. 

So how does one implement early stopping in practice? Surprisingly, the support for this (in tensorflow/tflearn) is slim. If you try digging around on how to use tensorflow for this, you'll likely find code snippets such as the following.

```python
validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
                      "precision": tf.contrib.metrics.streaming_precision,
                      "recall": tf.contrib.metrics.streaming_recall}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    #metrics=validation_metrics,
    early_stopping_metric='loss',
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

tf.logging.set_verbosity(tf.logging.ERROR)
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,
               monitors=[validation_monitor])
```




