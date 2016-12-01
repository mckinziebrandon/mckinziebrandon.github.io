---
layout: post
title:  "TensorFlow Support For Windows!"
date:   2016-11-30
excerpt: "Alas, the day has finally come. Goodbye Docker. Goodbye VMWare (well, sort of)."
feature: http://i.imgur.com/XkJom03.png
tag:
- tensorflow
- python
- machine learning
comments: false
---

# Easy as <kbd>p</kbd><kbd>i</kbd><kbd>p</kbd>

Time to once again thank <s>our overlords</s> the great software engineers at Google. Yesterday, posted on [the google developers blog](https://developers.googleblog.com/2016/11/tensorflow-0-12-adds-support-for-windows.html), was the announcement that TensorFlow r0.12 can be installed natively in Windows via

```bash
pip install tensorflow
```

Note that you can also install the tflearn module via pip! Of course, I had to wrestle with dependency issues for a bit to get this to work. Due in large part to the helpful folks on [this thread](https://www.reddit.com/r/MachineLearning/comments/5fk27q/n_tensorflow_012_adds_support_for_windows/), the following setup should ensure you avoid such issues:

* Safest bet appears to be using Python 3.3 or newer, although there are mixed reports of users getting this to work with Python 2.7 as well (I'm using 3.5). 
* Conda vs. Pip. I've always loved using Anaconda as my python package manager, but I ended up going back to pip for this release. Again, there seem to be mixed reviews about using Conda with this update. Personally, I prefer the path of least resistance when installing third-party software in Windows, so pip it is!
* Don't forget to update your environment variables (PATH) as needed.

Now, as I sit here typing this post in a Linux VM, I can't help but wish the developers at __Jekyll__ would release an official Windows build . . . 

