
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

BIM requires the following function, which clips pixel values of an image $X^\prime$:

Similar to the fast method we update the pixel values:

\begin{equation}
\tag{2.1}
\widetilde{X}_{n+1} = Clip_{X, \epsilon} \{ \widetilde{X}_{n} + \alpha sign(\nabla_{X} J(\widetilde{X}_{n}, Y_{true})) \}
\end{equation}

Here, the clip function is defined as:

\begin{equation}
\tag{2.2}
Clip_{X, \epsilon} \{ X^\prime \} (x, y, z) = min\{ 255, X(x, y, z) + \epsilon, max\{0, X(x, y, z)-\epsilon, X^\prime(x, y, z) \} \}
\end{equation}

For the implementation we initialize:

\begin{equation}
\tag{2.3}
\widetilde{X}_{0} = X
\end{equation}

For the implementation we do for the number of iterations `n`:

a) Compute $ X^\prime = \widetilde{X}_{n} + \alpha sign(\nabla_{X} J(\widetilde{X}_{n}, Y_{true}))$ as used in equation (2.1)

b) Compute $ X(x, y, z) + \epsilon $ and $ X(x, y, z) - \epsilon $

c) Evaluate equation (2.2) using steps a) and b)

$max\{0, X(x, y, z)-\epsilon, X^\prime(x, y, z) \}$ (?)

d) Retrieve updated adversarial image $\widetilde{X}_{1}$ as given in equation (2.1)

Values for the hyper parameters given in [2]:

- $\alpha = 1$

- Number of iterations: $min(4+\epsilon, 1.25 \cdot \epsilon)$

- $\epsilon $



### Iterative Least Likely Class Method


The methods mentioned above increase the cost of the correct class. This works well as long as there are few, well separated classes. For classifier with a lot of similar classes however, this can lead to uninteresting examples [5]. The Iterative Least-Likely Class overcomes this by maximizig the likelihood of the class with the lowest probability $Y_{L,L}$. This class is usually dissimilar from the correct class.

Similar to the BIM, we define a clip function:

\begin{equation}
\tag{3.1}
\widetilde{X}_{n+1} = Clip_{X, \epsilon} \{ \widetilde{X}_{n} - \alpha sign(\nabla_{X} J(\widetilde{X}_{n}, Y_{L,L})) \}
\end{equation}


As in (2.2) the Clip function is defined as:

\begin{equation}
\tag{3.2}
Clip_{X, \epsilon} \{ X^\prime \} (x, y, z) = min\{ 255, X(x, y, z) + \epsilon, max\{0, X(x, y, z)-\epsilon, X^\prime(x, y, z) \} \}
\end{equation}

For the implementation we initialize:

\begin{equation}
\tag{3.3}
\widetilde{X}_{0} = X
\end{equation}


### Deep Fool


