---
layout: post
title:  "Condensed Tutorials 1 - Conv Nets"
date:   2016-12-21
excerpt: "My notes on two tutorials from Colah's blog, condensed to the main ideas."
tags:
- condensed tutorial
- colah
comments: false
---


# Conv Nets: A Modular Perspective

[From this post on Colah's Blog](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/).

The title is inspired by the following figure. Colah mentions how groups of neurons, like \\(A\\), that appear in multiple places are sometimes called \textbf{modules}, and networks that use them are sometimes called modular neural networks. You can feed the output of one convolutional layer into another. With each layer, the network can detect higher-level, more abstract features.

<img src="{{site.url}}/assets/img/colah/ColahConv2.PNG" style="width: 300px;"/>

* Function of the \\(A\\) neurons: compute certain _features_.
* Max pooling layers: kind of ``zoom out''. They allow later convolutional layers to work on larger sections of the data. They also make us invariant to some very small transformations of the data.

------------------------------------------------------

# Understanding Convolutions

From the [subsequent tutorial](https://colah.github.io/posts/2014-07-Understanding-Convolutions/) on Colah's Blog.

## Example: Dropping a Ball

Imagine we drop a ball from some height onto the ground, where it only has one dimension of motion. How likely is it that a ball will go a distance c if you drop it and then drop it again from above the point at which it landed?
{: .notice}

{% capture images %}
    {{site.url}}/assets/img/colah/ColahBall.PNG
    {{site.url}}/assets/img/colah/ColahBall3.PNG
{% endcapture %}
{% include gallery images=images caption="Dropping a Ball. 1-D (left) and 2-d (right)." cols=2 %}


From basic probability, we know the result is a sum over possible outcomes, constrained by \\(a + b = c\\). It turns out this is actually the definition of the convolution of \\(f\\) and \\(g\\). 


\\[ \mathrm{Pr(a + b = c)} = \sum_{a + b = c} f(a) \cdot g(b) \\]
\\[ (f * g)(c) = \sum_{a + b = c} f(a) \cdot g(b) = \sum_a f(a) \cdot g(c - a) \\]


## Visualizing Convolutions

Keeping the same example in the back of our heads, consider a few interesting facts. Flipping directions: If \\(f(x)\\) yields the probability of landing a distance \\(x\\) away from where it was dropped, what about the probability that it was dropped a distance \\(x\\) from where it \textit{landed}? It is \\(f(-x)\\). 
	
![](http://colah.github.io/posts/2014-07-Understanding-Convolutions/img/ProbConv-Intermediate.png){: .med}

	
Above is a visualization of one term in the summation of \\((f * g)(c)\\). It is meant to show how we can move the bottom around to think about evaluating the convolution for different \\(c\\) values. We can relate these ideas to image recognition. Below are two common kernels used to convolve images with. 


* For _blurring_ images, we take simple (uniform) averages. ![]({{site.url}}/assets/img/colah/ColahKernel1.PNG){: .image-right .small}To relate to our ball example, this is like letting \\(f\\) be pixel intensity and \\(g\\) be pixel weight (here this is just 1/9). 



* For _edge detection_, we take the difference between two pixels. ![]({{site.url}}/assets/img/colah/ColahKernel2.PNG){: .image-right .small}This difference will be largest at edges, and essentially zero for similar-looking pixels. 




