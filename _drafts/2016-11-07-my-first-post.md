---
layout: post
title:  "My First Post"
date:   2016-11-07
excerpt: "Me testing out the functionality of Jekyll."
tags: [markdown,  jekyll, test]
feature: http://i.imgur.com/eb6g2Lk.jpg
comments: false
---

# Learning New Software

I find myself doing this often. One good example is right now. The Jekyll documentation is decent . . . I suppose. Nonetheless, it
is not "obvious" by any means how this all works together, nor is it clear to me where to find much of the answers. My approach:
just keep typing stuff in different files until things start to work. In this post, let's take a journey through my confusion!

### What is all this stuff?

![pic]({{site.url}}/assets/img/butterRobot.jpg){: .image-right height="10px"}

I *was* going to show a snippet of code below of some confusing jekyll-infused html, but apparently jekyll reads the snippet and
actually tries executing it. Sorry to let you, my imaginary reader, down. Hopefully this code snippet below that came with the
theme will cheer you up. <br/>


Wait, I need to type just a little more text to provide room for the code snippet. Of course, I could customize the appropriate
CSS file to resize the image, or I could even go brute force and just write the html img element in the poor markdown file. But we
all know that is not okay. Thus, I type. 

{% highlight html %}
<a href="#" class="btn btn-success">Success Button</a>
{% endhighlight %}

### Blockquotes

> On the internet, no one knows you're a robot that passes butter.

### Some of my Current Projects

1. This website.


### Picture of Dogs

<div markdown="0"><a href="{{site.url}}/assets/img/doggo.jpg" class="btn">Dog</a></div>
<div markdown="0"><a href="{{site.url}}/assets/img/doggo2.jpg" class="btn btn-success">Dog</a></div>
<div markdown="0"><a href="{{site.url}}/assets/img/doggo3.jpg" class="btn btn-warning">Dog</a></div>
<div markdown="0"><a href="{{site.url}}/assets/img/doggo4.jpg" class="btn btn-danger">Dog</a></div>
<div markdown="0"><a href="{{site.url}}/assets/img/tiger_one.jpg" class="btn btn-info">Cat?</a></div>

**TODO** Write more amazing posts.
{: .notice}
