
---

layout: page
title: State of the art
permalink: /state_of_the_art/
---


An adversarial example is an input with slight modifications that misleads the machine learning model. In the context of this project we focus on images as inputs. However, adversarial examples exists for other domains as well [1]. The perturbations are designed to be so small that they are hardly perceptible by a human or may not even be representable in an 8bit image if the model excepts 32bit inputs. To craft malicious inputs, a clean input is used and modified in a way that the network’s loss o this prediction increases while keeping the confidence high. To do so access to the model’s parameters is no necessary. [2] show that networks can also be fooled when presented with malicious images through cameras. With the adoption of neural networks in autonomous vehicles for example the existence of adversarial examples can cause serious safety concerns and has to be addressed. Since the discovery of their existence in 2013 [3] research has focussed on understanding their origin and how to defend against them. As of this writing defending against adversarial examples remains difficult. The best mechanism is adversarial training [4].

### History
In 2014 [3] showed that a neural net classifier can be fooled by a small modification to the input image. This perturbation is found by maximizing the network's prediction error. Moreover, a modified image which fools one classifier often can also fool other classifier to make the same mistake. This property is called *transferability* and allows black-box attacks (see below). As a reason for the existence of such adversarial examples the authors suggested the highly non-linearity of neural networks.

In 2015 this hypothesis was rejected. Instead, [5] suggest that instead the linear behaviour of networks (use of ReLUs, linearization of sigmoids around 0, etc. ) enables adversarial examples and introduced the *Fast Gradient Sign Method* (FGSM) to generate them. This non-iterative method generates examples in one step (thus fast) and leads to robust malicious images [2]. It takes one step of given size into the direction of increased loss by using the model’s gradient. One downside of the FGSM is that the manipulated images are often perceptible for humans. 

Iterative methods are slower but produce more subtle perturbation to images. One such method, which is derived from the FGSM, is the *Basic Iterative Method* (BIM). This extension of the former applies FGSM multiple times and clips values after each step to ensure that the pixel values stay similar to the original ones [2].

Both of the previous methods only try (they don’t guarantee a false classification) to change the prediction to a different class. When attacking a classifier with a lot of similar classes this can lead to uninteresting results. For example one dog breed would be classified as another dog breed but not as a say a cat. The *Iterative Least Likely Class Method* (ILLM) looks at the prediction on a clean image and modifies it to output the least likely class [2].

[6] Deepfool present an algorithm to …


The methods above modify all pixels to increase the loss. [7] propose that rather than modifying all pixels slightly, only one (or few) pixels can be modified with greater magnitude. The result is an overall less perturbed image. The modification is achieved by differential evolution.

The truly targeted method by [8] also does not modify all pixels and is distinct form the other methods in another way. Instead of generating adversarial examples from outputs the authors introduce a method which crafts an adversarial example from an input using the forward gradient of the network. This is achieved by iteratively computing saliency maps. From these, a specified number of features with the highest impact when perturbed are selected and adjusted according to a specified amount. This procedure stops when either the sample is classified as the desired target, the perturbation reaches a defined maximum or the maximum number of iterations is reached. This method does not modify all pixels. The results are highly effective, truly targeted as specified by the user and hardly perceptible for humans.

In addition to the perceptibility of the perturbations it has been shown that adversarial examples can even fool humans when only briefly exposed to the image [9]. This is significant since it could suggest transferability to the human brain.

In the following selected aforementioned attack methods are described in more detail.


### Attack Methods
Existing attack methods can be grouped into the following categories.

**White box**: The attacker has full access to the model with all it's parameters.

**Black box with probing**: An attacker has no access to the model's parameters. However, the model can be be queried to approximate the gradients.


**Black box without probing**: Here, the attacker has neither access nor can he query the model.

**Digital attack**: An attacker has direct access to digital data fed into the model.

Moreover, attacks can be **targeted** or **untargeted**. In the latter scenario the attack is successful if any wrong class is predicted. For the former attack type, a specific class is predicted. Most existing attack method require a gradient to work with. That's why in the case of black box attacks it is common to approximate the gradient by taking advantage of transferability.

The following are the methods that we explore in this project.

#### FGSM
This method computes a step of gradient descent and moves one step of magnitude $$\epsilon$$ into the direction of this gradient:

\begin{equation}
\tag{1.1}
\widetilde{x} = x + \eta
\end{equation}

\begin{equation}
\tag{1.2}
\eta = \epsilon \cdot sign(\nabla_{x} J(\Theta, x, y))
\end{equation}


#### Basic Iterative Method (BIM)
An extension of the FGSM is BIM. It applies the FGSM multiple times to an image with step size $$\alpha$$ and clips the results. This ensures the result is in the $$\epsilon$$ neighbourhood of the original image $$X$$.

The steps are:

Initialize image with the clean image for iteration $$N=0$$

\begin{equation}
\tag{2.1}
\widetilde{X}_{0} = X 
\end{equation}

Compute fast step:

\begin{equation}
\tag{2.2}
X^{\prime}\_{1} = \widetilde{X}\_{0} + \alpha sign(\nabla\_{X} J(\widetilde{X}\_{0}, Y\_{true}))
\end{equation}

Clip pixel values:

\begin{equation}
\tag{2.3}
\widetilde{X}\_{1} = min \( 255, X + \epsilon, max \( 0, X-\epsilon, X^{\prime}\_{1} \)\)
\end{equation}

Repeat for $$N=1$$

For the hyperparameter [2] suggest:

- $$\alpha$$ = 1
- Number of iterations: $$min(4+\epsilon, 1.25 \cdot \epsilon)$$


#### Iterative Least Likely Method (ILLM)
The ILLM manipulates an image to make the model predict the class which has the lowest probability in the clean case.

Similar to the BIM the steps are:

Initialize image with the clean image for iteration $$N=0$$

\begin{equation}
\tag{3.1}
\widetilde{X}_{0} = X 
\end{equation}

Compute fast step:

\begin{equation}
\tag{3.2}
X^{\prime}\_{1} = \widetilde{X}\_{0} - \alpha sign(\nabla\_{X} J(\widetilde{X}\_{0}, Y\_{LL}))
\end{equation}

Clip pixel values:

\begin{equation}
\tag{3.3}
\widetilde{X}\_{1} = min \( 255, X + \epsilon, max \( 0, X-\epsilon, X^{\prime}\_{1} \)\)
\end{equation}

Repeat for $$N=1$$



#### Deep Fool



### Defenses
As mentioned above, the best defence against adversarial examples is including them in the training of the model. [9] created a library to support this by providing the common attack methods.




--------
### References

[1] &emsp; Qin, Y., Carlini, N., Goodfellow, I., Cottrell, G., & Raffel, C. (n.d.). Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition. 12.

[2] &emsp; Kurakin, A., Goodfellow, I., & Bengio, S. (2017). Adversarial examples in the physical world. [ArXiv:1607.02533](http://arxiv.org/abs/1607.02533) [Cs, Stat].

[3] &emsp; Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. [ArXiv:1312.6199](http://arxiv.org/abs/1312.6199) [Cs].

[4] &emsp; Goodfellow, I., Papernot, N., Huang, S., Duan, R., Abeel, P., & Clark, J. (2017). Attacking Machine Learning with Adversarial Examples. [https://openai.com/blog/adversarial-example-research/](https://openai.com/blog/adversarial-example-research/)

[5] &emsp; Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. [ArXiv:1412.6572](http://arxiv.org/abs/1412.6572) [Cs, Stat].

[6] &emsp; Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016). DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [https://doi.org/10.1109/cvpr.2016.282](https://doi.org/10.1109/cvpr.2016.282)

[7] &emsp; Su, J., Vargas, D. V., & Kouichi, S. (2019). One pixel attack for fooling deep neural networks. IEEE Transactions on Evolutionary Computation, 23(5), 828–841. [https://doi.org/10.1109/TEVC.2019.2890858](https://doi.org/10.1109/TEVC.2019.2890858)

[8] &emsp; Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2015). The Limitations of Deep Learning in Adversarial Settings. ArXiv:1511.07528 [Cs, Stat]. [http://arxiv.org/abs/1511.07528](http://arxiv.org/abs/1511.07528])


[9] &emsp; Elsayed, G. F., Shankar, S., Cheung, B., Papernot, N., Kurakin, A., Goodfellow, I., & Sohl-Dickstein, J. (2018). Adversarial Examples that Fool both Computer Vision and Time-Limited Humans. ArXiv:1802.08195 [Cs, q-Bio, Stat]. [http://arxiv.org/abs/1802.08195](http://arxiv.org/abs/1802.08195)

[10] &emsp; Papernot, N., Faghri, F., Carlini, N., Goodfellow, I., Feinman, R., Kurakin, A., Xie, C., Sharma, Y., Brown, T., Roy, A., Matyasko, A., Behzadan, V., Hambardzumyan, K., Zhang, Z., Juang, Y.-L., Li, Z., Sheatsley, R., Garg, A., Uesato, J., … McDaniel, P. (2018). Technical Report on the CleverHans v2.1.0 Adversarial Examples Library. ArXiv:1610.00768 [Cs, Stat]. [http://arxiv.org/abs/1610.00768](http://arxiv.org/abs/1610.00768)