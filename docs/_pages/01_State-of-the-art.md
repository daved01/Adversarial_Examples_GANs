---

layout: page
title: State of the art
permalink: /state_of_the_art/
---
-> Explain history


Deep neural networks are used for a variety of applications. With a broader adoption of critical functions like self driving cars or face recognition based security systems the safety and robustness of the networks becomes more important than ever.

In 2014 [1] found that neural networks can easily be fooled by slight modifications to their inputs. The authors considered the high expressivness of the networks to be the reason for this phenomenon. Later, [2] show that it is rather the linear behaviour of the networks that play a role.

...

Neural networks consistently misclassify intentionally perturbed inputs [2]. The perturbations can be applied in such a way that the network predicts a different class with high confidence. To the human eye however, the perturbations are, up to a certrain level, barely perceptible [2]. Inputs manipulated with the goal of being classified as a different class are called *adversarial examples* [???]. As shown in [1, 2], the same example gets falsely classified as the same class by different networks. This is called transferability.

[3] show that the perturbations persist even through a camera. The noise introduced by using a phone camera does not destroy the effect.


## Types of Attacks
-> Structure attack methods


## Attack Methods

Since the discovery of the existence of adversarial examples in [1] methods have been developed which can be grouped into the following categories [4].

**White box**

Attacker has full access to the model with all it's parameters.

**Black box with probing**

Attacker has no access do model's parameters. However, the model can be be querried to approximate the gradients.

**Black box without probing**

Here, the attacker has neither access nor can he querry the model.

**Digital attack**

Attacker has direct access to digital data fed into the model.

Moreover, attacks can be **targeted** or **untargeted**. In the latter scenario the attack is succesfull if any wrong class is predicted. For the former attack, a specific class is predicted.

Most (all?) exsting attack method require a gradient to work with. That's why in the case of black box attacks it is common to approximate the gradient by taking advantage of transferability.

The following are the methods that we explore in this project.


### Fast Gradient Sign Method

This method by [2] generates adversarial examples quickly. It computes a step of gradient descent and moves one step of magnitude $\epsilon$ into the direction of this gradient:

\begin{equation}
\tag{1.1}
\widetilde{x} = x + \eta
\end{equation}

\begin{equation}
\tag{1.2}
\eta = \epsilon \cdot sign(\nabla_{x} J(\Theta, x, y))
\end{equation}


### Basic Iterative Method


### Iterative Least Likely Class Method


### Deep Fool


