---
layout: post
title:  "No-Fluff Reinforcement Learning"
date:   2020-01-19
excerpt: "Synthesis of Reinforcement Learning Principles & Techniques"
tags: [reinforcement learning]
comments: true
---

# Brief Overview. 

I've taken a few courses that introduce reinforcement learning, and they all do so in a way that feels convoluted/hard to retain (for me). There are many overlapping concepts introduced together, and figuring out what's related to what/why/when/how can become tricky. Here, I'll give the bare-bones no-nonsense overview of RL. I won't do the traditional exhaustive introduction of MDPs and delineation of model-based vs model-free, on-policy, off-policy, etc. I'll stick to the main concepts that have stood the test of time and are still a componenet of more sophisticated modern techniques.

# Definitions

## World Model

Although not strictly a part of the definition of reinforcement learning (TODO: verify/confirm), the most common RL approaches model the world as a **Markov Decision Process** (MDP). The components of an MDP explicitly used across most RL approaches:

* Set of states \\(S\\)
* \\(s_{start} \in S\\)
* Possible actions from state \\(s\\): Actions(\\(s\\))
* IsEnd(\\(s\\))
* Discount factor \\(0 \le \gamma \le 1\\). 
                                       
Note that MDPs also include a *transition function* \\(T(s, a, s') \triangleq Pr(s' \mid a, s)\\) and *reward function* \\(R(s, a, s')\\). However, in RL we don't know these ahead of time. As we'll see later, some RL approaches try to model these quantities directly (model-based), while others learn higher-level quantities like *expected utility* \\(Q\\). 
{: .notice}

## Data and Training

Our data consists of one or more **episodes** \\(\{s_{start}, (a_1, r_1, s_1), (a_2, r_2, s_2), \ldots, s_{end}\}\\). In contrast with supervised learning, in RL the model generates the training data itself. It does so via the following high-level algorithm:

{% highlight python %} 
def rl_template(s_start, num_steps):
    s_prev = s_start
    for t in range(num_steps):
        a_t = policy(s_prev)
        r_t, s_t = take_action(a_t)
        update_params(...)
        s_prev = s_t
{% endhighlight %}

------------------------------------------

# Learning to Act

So, what are we learning here? What's the ultimate goal? **We want to learn an optimal policy \\(\pi: s \mapsto a\\)**. In other words, the goal of RL is to learn how an agent should act within its world. The interesting part, again, arises from the fact that, in order to collect training data, our agent must explore the world and try out some actions. Stated another way: in order to learn how to act, we must act! Before learning *how* to act, we must first decide how to *evaluate* whether some policy \\(\pi_a\\) is "better" than another policy \\( \pi_b \\).

## Expected Utility

How do we decide/learn what actions to take? Naively, one might guess we should take whatever actions result in the highest reward. We'll take that a step further and say *we should take actions that have the highest **expected** reward*. Formally, for a given episode, define the **utility** at timestep t as

\\[ u_t = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \cdots \\]

which is a *random quantity* due to the stochastic nature of the [unknown] transitions \\( s \rightarrow a \rightarrow s' \\). A common approach ("Model-Free Monte Carlo") is to learn/estimate the **expected utility** of taking an action \\(a\\) from state \\(s\\), and then following some policy \\(\pi\\), often referred to as the Q-value \\(Q_\pi(s, a)\\). We denote our estimate of \\(Q\\) as \\(\hat Q\\). There are a few equivalent ways we can compute \\(\hat Q\\):

\\[ \hat Q_{\pi}(s, a) = \text{Average}( u_t[s_{t-1}{=}s, a_t{=}a] ) \\]
\\[ \hat Q_{\pi}(s, a) \leftarrow  (1 - \eta) \hat Q_{\pi}(s, a) + \eta u  \\]
\\[ \hat Q_{\pi}(s, a) \leftarrow  \hat Q_{\pi}(s, a)  - \eta ( \hat Q_{\pi}(s, a) - u  )  \\]
where \\(  \eta := \frac{1}{1+\text{NumUpdates}(s, a)} \\). We refer to these as the average, convex combination, and stochastic gradient formulations, respectively.

The problem here is that we are evaluating how good an action is based on some fixed policy \\(\pi\\), which may not even be that great of a policy. Can we estimate the true expected utility instead? Yes, that's what algorithms like **Q-learning** do. Q-learning is defined by the following update rule, for a given path \\((s, a, r, s')\\):

\\[ \hat Q_{opt}(s, a) \leftarrow (1 - \eta) \hat Q_{opt}(s, a) + \eta (r + \gamma \max_{a'} \hat Q_{opt}(s', a'))  \\]

which tells us the expected utility of taking an action from a given state, independent of some fixed/specified policy. It's a greedy estimate though, since we don't play out the full episode (required to compute \\(u_t\\)) -- we approximate \\(u_t\\) using the immediate reward and the current estimate for \\(Q(s', a')\\). One problem with this formulation is that it treats the state/action space as disjoint black boxes, which results in poor generalization ability. As usual, we can aid generalizeabilty by incorporating features \\(\mathbf{\phi}\\) and weights \\( \mathbf{w} \\):

\\[ \hat Q_{opt}(s, a; \mathbf{w}) := \mathbf{w}^T \mathbf{\phi}(s, a) \\]
\\[ \mathbf{w} \leftarrow \mathbf{w} - \eta \bigg[ \hat Q_{opt}(s, a; \mathbf{w}) - (r + \gamma \max_{a'} \hat Q_{opt}(s', a')) \bigg] \mathbf{\phi}(s, a) \\]

which is referred to as **Q-learning with function approximation**.

## Taking Action

Many common approaches for \\(\pi\\)are actually rather primitive/simple. They typically do some balance of **exploitation**, \\( \pi(s) := \text{arg}\max_a \hat Q_{opt}(s, a) \\), and **exploration**, \\(  \pi(s) := \text{Random(Actions(s))} \\).


