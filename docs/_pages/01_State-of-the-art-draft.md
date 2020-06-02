

An adversarial example is an input with slight modifications that misleads the machine learning model. In the context of this project we focus on images as inputs. However, adversarial examples exists for other domains as well [???]. The perturbations are designed to be so small that they are hardly perceptible by a human or may not even be representable in an 8bit image if the model excepts 32bit inputs. To craft malicious inputs, a clean input is used and modified in a way that the network’s loss o this prediction increases while keeping the confidence high. To do so access to the model’s parameters is no necessary. [2] show that networks can also be fooled when presented with malicious images through cameras. With the adoption of neural networks in autonomous vehicles for example the existence of adversarial examples can cause serious safety concerns and has to be addressed. Since the discovery of their existence in 2013 [3] research has focussed on understanding their origin and how to defend against them. As of this writing defending against adversarial examples remains difficult. The best mechanism is adversarial training [4].

### History
In 2014 [3] showed that a neural net classifier can be fooled by a small modification to the input image. This perturbation is found by maximizing the network's prediction error. Moreover, a modified image which fools one classifier often can also fool other classifier to make the same mistake. This property is called *transferability* and allows black-box attacks (see below). As a reason for the existence of such adversarial examples the authors suggested the highly non-linearity of neural networks.

In 2015 this hypothesis was rejected. Instead, [5] suggest that instead the linear behaviour of networks (use of ReLUs, linearization of sigmoids around 0, etc. ) enables adversarial examples and introduced the *Fast Gradient Sign Method* (FGSM) to generate them. This non-iterative method generates examples in one step (thus fast) and leads to robust malicious images [2]. It takes one step of given size into the direction of increased loss by using the model’s gradient. One downside of the FGSM is that the manipulated images are often perceptible for humans. 

Iterative methods are slower but produce more subtle perturbation to images. One such method, which is derived from the FGSM, is the *Basic Iterative Method* (BIM). This extension of the former applies FGSM multiple times and clips values after each step to ensure that the pixel values stay similar to the original ones [2].

Both of the previous methods only try (they don’t guarantee a false classification) to change the prediction to a different class. When attacking a classifier with a lot of similar classes this can lead to uninteresting results. For example one dog breed would be classified as another dog breed but not as a say a cat. The *Iterative Least Likely Class Method* (ILLM) looks at the prediction on a clean image and modifies it to output the least likely class [2].

[6] Deepfool present an algorithm to …

The methods above all modify all pixels to increase the loss. [7] propose that rather than modifying all pixels slightly, only one (or few) pixels can be modified with greater magnitude. The result is an overall less perturbed image. The modification is achieved by differential evolution.

It has been shown that adversarial examples can even fool humans when only briefly exposed to the image [8]. This is significant since it could suggest transferability to the human brain.

In the following the aforementioned attack methods are described in more detail.


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



#### Iterative Least Likely Method (ILLM)



#### Deep Fool



### Defenses
As mentioned above, the best defence against adversarial examples is including them in the training of the model. [9] created a library to support this by providing the common attack methods.




--------
### References

[1]

[2] &emsp; Kurakin, A., Goodfellow, I., & Bengio, S. (2017). Adversarial examples in the physical world. [ArXiv:1607.02533](http://arxiv.org/abs/1607.02533) [Cs, Stat].

[3] &emsp; Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. [ArXiv:1312.6199](http://arxiv.org/abs/1312.6199) [Cs].

[4] &emsp; Goodfellow, I., Papernot, N., Huang, S., Duan, R., Abeel, P., & Clark, J. (2017). Attacking Machine Learning with Adversarial Examples. [https://openai.com/blog/adversarial-example-research/](https://openai.com/blog/adversarial-example-research/)

[5] &emsp; Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. [ArXiv:1412.6572](http://arxiv.org/abs/1412.6572) [Cs, Stat].

[6] &emsp; Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016). DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [https://doi.org/10.1109/cvpr.2016.282](https://doi.org/10.1109/cvpr.2016.282)

[7] &emsp; Su, J., Vargas, D. V., & Kouichi, S. (2019). One pixel attack for fooling deep neural networks. IEEE Transactions on Evolutionary Computation, 23(5), 828–841. [https://doi.org/10.1109/TEVC.2019.2890858](https://doi.org/10.1109/TEVC.2019.2890858)

[8] &emsp; Elsayed, G. F., Shankar, S., Cheung, B., Papernot, N., Kurakin, A., Goodfellow, I., & Sohl-Dickstein, J. (2018). Adversarial Examples that Fool both Computer Vision and Time-Limited Humans. ArXiv:1802.08195 [Cs, q-Bio, Stat]. [http://arxiv.org/abs/1802.08195](http://arxiv.org/abs/1802.08195)

[9] &emsp; Papernot, N., Faghri, F., Carlini, N., Goodfellow, I., Feinman, R., Kurakin, A., Xie, C., Sharma, Y., Brown, T., Roy, A., Matyasko, A., Behzadan, V., Hambardzumyan, K., Zhang, Z., Juang, Y.-L., Li, Z., Sheatsley, R., Garg, A., Uesato, J., … McDaniel, P. (2018). Technical Report on the CleverHans v2.1.0 Adversarial Examples Library. ArXiv:1610.00768 [Cs, Stat]. [http://arxiv.org/abs/1610.00768](http://arxiv.org/abs/1610.00768)
