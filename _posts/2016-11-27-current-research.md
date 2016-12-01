---
layout: post
title:  "Automatic Architecture Generation for Deep Neural Networks"
date:   2016-11-27
excerpt: "A brief overview of my current research."
research: true
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


My current research, led by Professor Dawn Song's research group at UC Berkeley, aims to solve this problem. Rather than having humans tinker around with designing deep neural networks, this project aims to write software that can learn to build deep networks on its own. In some sense, its learning process is akin to a child building structures with lego blocks, trying to figure out which combinations are 'best' and which combinations should be avoided altogether. 


### Automatic Code Generation

The project itself is primarily written in the Scala programming language, and it outputs ready-to-run tensorflow  (python) implementations. The specifics of any given network are abstracted away. Instead, the networks are described as consisting of generic *components*. For the sake of brevity, we can think of these components as the essential building blocks of any deep network. In particular, they give us *topological* information, i.e. how the network graph is shaped.

The process of code generation involves searching through many different network configurations and optimizing them for some pre-defined goal. In principle, this approach has the potential to eliminate time spent on building highly-specialized/domain-specific networks wherein we may not have particularly good reasons for the design decisions made/tested. Rather than saving specific network models that have performed well, we'd like to store models that can themselves build good models. Why save a single structure of legos when you can save progressively better lego builders?


### Summary of What Follows

For the remainder of this post, I'll describe more about my role in this project. In doing so, I'll attempt to provide some insight into how I approach research problems in general.


# Task 1: Early Stopping

Say you're a sentient machine trying to figure out the best network build. As we've discussed, you have about an infinite number of ways to approach this. Regardless of what you choose, you'll certainly need a way to *evaluate* how well you've done after constructing a given architecture.
Unfortunately, even for a fast AI such as yourself, training and evaluation can take a *really* long time. 

A common technique for speeding up the training and evaluation, known as **early stopping**, involving periodically "peeking under the hood" of the network as it learns, and determining whether or not (a) it is mostly done learning (negligible improvement), or (b) it has "learned enough," where it
is up to the designer to determine what "enough" means. 

So how does one implement early stopping in practice? Surprisingly, the support for this (in tensorflow/tflearn) is slim. If you try digging around on how to use tensorflow for this, you'll likely find code snippets such as the following.

{% highlight python %}
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
{% endhighlight %}

Unfortunately, a lot of these recommended approaches are outdated. Although I did manage to hack together a working implementation of early stopping with tf.contrib.learn (see TensorFlow Notebooks section), I had to suppress a lot of warning/info messages from TensorFlow (again, _a lot_) that were
solely due to me using certain methods of theirs (as recommended by their tutorials) that are now deprecated and confusingly don't seem to have direct replacements in new versions. But enough of that for now, let's see how we can use the tflearn library instead for a more reliable solution.  

### The Callback Class and TFLearn

The cleanest way I've implemented early stopping thus far has been with a little help from tflearn (distinct from tf.contrib.learn). Although they currently don't support early stopping explicitly, a nice workaround is defining a **callback** object. Here is a working proof-of-concept example below. But first, a brief overview. 


The following is a code snippet (comments mostly mine) directly from [__trainer.py__](https://github.com/tflearn/tflearn/blob/master/tflearn/helpers/trainer.py#L281) in the tflearn github repository, where I'm only showing the relevant parts/logic. 

{% highlight python %}
try:
    for epoch in range(n_epoch):
        # . . . Setup stuff for epoch here . . . 
        for batch_step in range(max_batches_len):
            # . . . Setup stuff for next batch here . . . 
            for i, train_op in enumerate(self.train_ops):
                caller.on_sub_batch_begin(self.training_state)
                # Train our model and store desired information in the train_op that
                # we (the user) pass to the trainer as an initialization argument.
                snapshot = train_op._train(self.training_state.step,
                                           (bool(self.best_checkpoint_path) | snapshot_epoch),
                                           snapshot_step,
                                           show_metric)
                # Update training state. The training state object tells us 
                # how our model is doing at various stages of training.
                self.training_state.update(train_op, train_ops_count)
            # All optimizers batch end
            self.session.run(self.incr_global_step)
            caller.on_batch_end(self.training_state, snapshot)
        # ---------- [What we care about] -------------
        # Epoch end. We define what on_epoch_end does. In this
        # case, I'll have it raise an exception if our validation accuracy
        # reaches some desired threshold. 
        caller.on_epoch_end(self.training_state)
        # ---------------------------------------------
finally:
    # Once we raise the exception, this code block will execute. 
    # Note only afterward will our catch block execute. 
    caller.on_train_end(self.training_state)
    for t in self.train_ops:
        t.train_dflow.interrupt()
    # Set back train_ops
    self.train_ops = original_train_ops
{% endhighlight %}


### Example Usage: Two-Layer Network for Digit Classification

To illustrate how we can accomplish early stopping with a callback class, let's setup an example. For those familiar with tensorflow, feel free to skip this small subsection. Below is how one can construct a simple two-layer (Input Layer '0' -> Hidden Layer '1' -> Output Layer '2') neural network for classifying handwritten digits (groundbreaking, I know). 


```python
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

n_features = 784
n_hidden = 256
n_classes = 10

# Define the inputs/outputs/weights as usual.
X = tf.placeholder("float", [None, n_features])
Y = tf.placeholder("float", [None, n_classes])

# Define the connections/weights and biases between layers.
W1 = tf.Variable(tf.random_normal([n_features, n_hidden]), name='W1')
W2 = tf.Variable(tf.random_normal([n_hidden, n_hidden]), name='W2')
W3 = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='W3')
b1 = tf.Variable(tf.random_normal([n_hidden]), name='b1')
b2 = tf.Variable(tf.random_normal([n_hidden]), name='b2')
b3 = tf.Variable(tf.random_normal([n_classes]), name='b3')

# Define the operations throughout the network.
net = tf.tanh(tf.add(tf.matmul(X, W1), b1))
net = tf.tanh(tf.add(tf.matmul(net, W2), b2))
net = tf.add(tf.matmul(net, W3), b3)

# Define the optimization problem.
loss      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
accuracy  = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1) ), tf.float32), name='acc')
```

### Define the TrainOp and Trainer Objects

As you may have noticed when reading the trainer.py snippet shown earlier, tflearn makes use of __tflearn.trainOp__ and __tflearn.Trainer__ objects. These allow use to cleanly specify how we want to define the import aspects of our training process. Here, I feed in the specifications define in the last section, namely 

* Compute loss with the cross-entropy function.
* Optimize with gradient descent.
* Define accuracy by fraction of predictions that match the labels in Y. 


```python
trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer, metric=accuracy, batch_size=128)
trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=1)
```

## Key Idea: EarlyStoppingCallback Class

I show a proof-of-concept version of early stopping below. This is the simplest possible case: just stop training after the first epoch no matter what. It is up to the user to decide the conditions they want to trigger the stopping on. The available tools for determining when training should end are given to us by a **training state** object. It stores the variables that one would expect: validation/training accuracies and losses, current epoch, etc. See [this jupyter notebook of mine on early stopping](http://mckinziebrandon.me/TensorflowNotebooks/2016/11/20/early-stopping.html) for more details. 


```python
class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        # Store a validation accuracy threshold, which we can compare against
        # the current validation accuracy at, say, each epoch, each batch step, etc.
        self.val_acc_thresh = val_acc_thresh
    
    def on_epoch_end(self, training_state):
        """ 
        This is the final method called in trainer.py in the epoch loop. 
        We can stop training and leave without losing any information with a simple exception.  
        """
        print("Terminating training at the end of epoch", training_state.epoch)
        raise StopIteration
    
    def on_train_end(self, training_state):
        """
        Furthermore, tflearn will then immediately call this method after we terminate training, 
        (or when training ends regardless). This would be a good time to store any additional 
        information that tflearn doesn't store already.
        """
        print("Successfully left training! Final model accuracy:", training_state.acc_value)
       
        
# Initialize our callback with desired accuracy threshold.  
early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.5)
```

## Result: Train the Model and Stop Early

Now we can pass our callback object to the trainer.fit function, sit back, and relax. Once our conditions are met for training termination, tflearn will leave the training loop, save the current state of our model, and enter our catch block, as shown below. 


```python
try:
    # Give it to our trainer and let it fit the data. 
    trainer.fit(feed_dicts={X: trainX, Y: trainY}, 
                val_feed_dicts={X: testX, Y: testY}, 
                n_epoch=1, 
                show_metric=True, # Calculate accuracy and display at every step.
                callbacks=early_stopping_cb)
except StopIteration:
    print("Caught callback exception. Returning control to user program.")
    
```

    Training Step: 860  | total loss: [1m[32m1.73372[0m[0m
    | Optimizer | epoch: 002 | loss: 1.73372 - acc: 0.8196 | val_loss: 1.87058 - val_acc: 0.8011 -- iter: 55000/55000
    Training Step: 860  | total loss: [1m[32m1.73372[0m[0m
    | Optimizer | epoch: 002 | loss: 1.73372 - acc: 0.8196 | val_loss: 1.87058 - val_acc: 0.8011 -- iter: 55000/55000
    --
    Terminating training at the end of epoch 2
    Successfully left training! Final model accuracy: 0.8196054697036743
    Caught callback exception. Returning control to user program.


## Alternative Implementation with TFLearn Layers

We just went through an example that combined tensorflow with tflearn together. Using helpers such as tflearn.trainOp and tflearn.Trainer allows one to implement more general networks by extending tensorflow with tflearn. However, we could go an even simpler route (but, of course, less flexible/generalizable) and solely use tflearn without explictly calling tensorflow functions at all. An example is shown below, early stopping included. 


```python
from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np
from sklearn.metrics import roc_auc_score
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

# Load the data and handle any preprocessing here.
X, Y, testX, testY = mnist.load_data(one_hot=True)
X, Y  = shuffle(X, Y)
X     = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Define our network architecture: a simple 2-layer network of the form
# InputImages -> Fully Connected -> Softmax
out_readin1          = input_data(shape=[None,28,28,1])
out_fully_connected2 = fully_connected(out_readin1, 10)
out_softmax3         = fully_connected(out_fully_connected2, 10, activation='softmax')

hash='f0c188c3777519fb93f1a825ca758a0c'
scriptid='MNIST-f0c188c3777519fb93f1a825ca758a0c'

# Define our training metrics. 
network = regression(out_softmax3, 
                     optimizer='adam', 
                     learning_rate=0.01, 
                     loss='categorical_crossentropy', 
                     name='target')

model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit(X, Y, 
          n_epoch=1, 
          validation_set=(testX, testY), 
          snapshot_step=10, 
          snapshot_epoch=False, 
          show_metric=True, 
          run_id=scriptid)


prediction = model.predict(testX)
auc=roc_auc_score(testY, prediction, average='macro', sample_weight=None)
accuracy=model.evaluate(testX,testY)

print("Accuracy:", accuracy)
print("ROC AUC Score:", auc)

```





