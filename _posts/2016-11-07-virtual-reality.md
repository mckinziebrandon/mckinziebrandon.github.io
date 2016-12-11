---
layout: post
title:  "Virtual Reality at MIT"
date:   2016-11-07
excerpt: "The journey of a young man learning to juggle in a virtual world."
research: true
feature: http://i.imgur.com/ffq5hlx.jpg
tag:
- research
- virtual reality
- Unity3D
- MIT
comments: true
---

# When Learning Is Too Dangerous

Have you ever dreamed of learning a new skill, but you were afraid of injury or exerting too much effort? That's how I felt when I used to dream about
juggling. Yes, tossing any imaginable assortment of objects into the air and be the commander of their trajectory. But alas, all
my years studying physics had ingrained in me a fear of accelerating objects and their devastating effects. I was afraid.

> . . . until I found *virtual reality*.

## Virtual Training Environments

A hot topic in the current VR research community is creating virtual spaces that allow one to freely experiment and learn new
skills. The options are without bound. My fear of large quantities of small objects falling from the sky prevented me from
exploring the world of juggling. Yet, as seen in the video below, virtual reality allowed me to learn in a safe virtual space. 
     
<iframe width="560" height="315" src="https://www.youtube.com/embed/si39l5rl_mU" frameborder="0"></iframe>


Here is what's going on:

__Hardware/Software Overview__: 

* This project was written in the Unity3D game engine. 
* The programs responsible for translating my movements were written in the C# programming language. The primary libraries used were Unity's framework (of course) and the SteamVR Unity plugin's library for interfacing with the HTC Vive. 
* The thing blocking my eyes is the __HTC Vive__, which allows me to enter virtual worlds after I create them.
* Attached to my arms are a total of four __Inertial Measurement Units (IMU).__ Their job is to provide spatial orientation information. Each arm has two IMU's and an __Arduino__.
* Each hand has __bend sensors__ on the fingers. (well, only one finger per hand in this video. That's what I get for only taking one video before the project was finished)

__Watch the Laptop Screen__:

* Even with the professional quality video, it may be hard to see what is going on. As far as I'm aware (in the video), I am a robot in a strange world of floating red balls. Furthermore, the red balls become green when a grab them with my new virtual
    hands. I'm able to pick them up and toss them around __without needing to hold a controller__. I can see my virtual body, my
    virtual arms, and my virtual hands move exactly as I do. I could not tell you the number of times I hit the walls of that
    office while juggling in my virtual world (enough). 

__Bonus Video__:

Below is a higher-quality demo of the project given by Andres Calvo, my advisor on the project and Master's student at the Media Lab. Thanks Andres!

<iframe width="560" height="315" src="https://www.youtube.com/embed/V7kX_EF_Fik" frameborder="0" allowfullscreen></iframe>


## Modeling Experience

{% capture images %}
    http://i.imgur.com/MJqBSs1.jpg
    http://i.imgur.com/fd73lOW.jpg 
    http://i.imgur.com/dx4xm93.jpg
    http://i.imgur.com/l8HTSub.jpg
{% endcapture %}
{% include gallery images=images caption="Professional Photoshoot" cols=3 %}

**Acknowledgments:** Shoutout to Andres Calvo and John Busche for making this project possible! Two of the best team members I've ever been fortunate enough to have. 
{: .notice}
