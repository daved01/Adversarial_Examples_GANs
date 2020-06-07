---

layout: page
title: Background
permalink: /state_of_the_art/
---

An adversarial example is a slightly modified input that is designed to mislead a machine learning model. In the context of this project we focus on images as inputs. However, adversarial examples exists for other domains as well [1]. To craft such malicious inputs, a clean input is used and modified in a way that the network’s loss on it increases while the confidence is kept high. The induced perturbations are designed to be so subtle that they are hardly perceptible to a human. [2] show that networks can even be fooled when presented with malicious images through cameras. 

With the adoption of neural networks in autonomous vehicles for example the existence of adversarial examples can cause serious safety concerns and has to be addressed. Since the discovery of their existence in 2013 [3] research has focussed on understanding their origin and how to defend against them.


## Why adversarial examples exist
Since the discovery that a neural net classifier can be fooled by a small modification to the input image [3], varying explanations for this phenomenon have been given. The first hypothesis was the highly non-linear nature of neural networks [3]. The authors also found that adversarial examples transfer between different models. This property is called *transferability*.

In 2015 this hypothesis was rejected. Instead, [4] suggest that the linear behaviour of networks (use of Rectified Linear Units, linear behaviour of sigmoids around 0, etc.) enables adversarial examples and introduced the *Fast Gradient Sign Method* (FGSM) to generate them.

Since then multiple attack methods based on this assumption of the linear behaviour have been developed. We present some of them below.

Recently the focus in the search of the origin of adversarial examples has shifted away from the networks to the data. [5, 6] claim that the existence of non-robust features in the data enables adversarial examples. These are features which are weakly correlated with the predicted class, but the classifier relays strongly on them. An adversary anti-correlates them which can shift the prediction. It is possible to train a robust classifier by using robust features only. These features are strongly correlated and not vulnerable to attacks. The authors show that robust features are features which gradient's are identifiable for humans. The cost of achieving high adversarial robustness is a lower accuracy. Intuitively this is because the classifier relays on fewer features. Interestingly, a robust model is also more interpretable to humans. And related to GANs, which we cover in the next project. [6] further support the hypothesis that adversarial examples are based in the data. They show a model trained on adversarial data only can still yield a good accuracy, supporting the earlier claim that model relays on imperceptible features. In a second experiment they showed that using a robust dataset can transfer robustness to different networks. Their findings are discussed in this [article](https://distill.pub/2019/advex-bugs-discussion/).


## Existing attack methods and how to generate them
Existing attack methods can be grouped into the following categories.

**White box**: The attacker has full access to the model with all its parameters.

**Black box with probing**: An attacker has no access to the model's parameters. However, the model can be be queried to approximate the gradients.

**Black box without probing**: Here, the attacker has neither access nor can he query the model.

**Digital attack**: An attacker has direct access to digital data fed into the model.

Recently, [7] have shown that network parameter can be extracted by analyzing the power consumption of the model during inference, a method called differential power analysis.

Moreover, attacks can be **targeted** or **untargeted**. In the latter scenario the attack is successful if any wrong class is predicted. For the former attack type, a specific class is targeted. Most existing attack method require a gradient to work with. That's why in the case of black box attacks it is common to approximate the gradient by taking advantage of transferability.

The following are the methods that we explore in this project.

### FGSM
The Fast Gradient Sign Method was introduced in 2015 by [4]. This non-iterative method generates examples in one step and leads to robust adversaries [2]. It computes a step of gradient descent and moves one step of magnitude $$\epsilon$$ into the direction of this gradient:

\begin{equation}
\tag{1.1}
\widetilde{X} = X + \eta
\end{equation}

\begin{equation}
\tag{1.2}
\eta = \epsilon sign(\nabla_{x} J(\Theta, x, y))
\end{equation}

One downside of the FGSM is that the manipulated images are often perceptible for humans. This can be improved by using iterative methods.


### Basic Iterative Method (BIM)
An extension of the FGSM is BIM. It applies the FGSM multiple times to an image with step size $$\alpha$$ and clips the resulting pixel values to ensure that they stay similar to the original ones [2].

Iterative methods like the BIM are slower but generally produce more subtle perturbation to images.

The steps are:

Initialize with the clean image $$X$$ for iteration $$N=0$$

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


### Iterative Least Likely Method (ILLM)
Both of the previous methods only try (they don’t guarantee a false classification) to change the prediction to a different class. When attacking a classifier with a lot of similar classes this can lead to uninteresting results. For example, one dog breed would be classified as another dog breed but not as a cat. The *Iterative Least Likely Class Method* (ILLM) looks at the prediction on a clean image and modifies it to output the least likely class [2].

Similar to the BIM the steps are:

Initialize with the clean image for iteration $$N=0$$

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


### Deep Fool
[8] Deepfool present an algorithm to …


### Targeted Papernot et al
The truly targeted method by [9] also does not modify all pixels and is distinct form the other methods in another way. Instead of generating adversarial examples from outputs the authors introduce a method which crafts an adversarial example from an input using the forward gradient of the network. This is achieved by iteratively computing saliency maps. From these, a specified number of features with the highest impact when perturbed are selected and adjusted according to a specified amount. This procedure stops when either the sample is classified as the desired target, the perturbation reaches a defined maximum or the maximum number of iterations is reached. This method does not modify all pixels. The results are highly effective, truly targeted as specified by the user and hardly perceptible for humans.


### Other methods
The methods above modify all pixels to increase the loss. [10] propose that rather than modifying all pixels slightly, only one (or few) pixels can be modified with greater magnitude. The result is an overall less perturbed image. The modification is generated by differential evolution.

In addition to the perceptibility of the perturbations it has been shown that adversarial examples can even fool humans when only briefly exposed to the image [11]. This is significant since it would suggest transferability to the human brain.


## How to defend against them
As of this writing the best defence against adversarial examples is to onclude them in the training data of the model (adversarial training). [12] created a library to support this by providing the common attack methods.


__________________

## References

[1] &emsp; Qin, Y., Carlini, N., Goodfellow, I., Cottrell, G., & Raffel, C. (n.d.). Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition. 12.

[2] &emsp; Kurakin, A., Goodfellow, I., & Bengio, S. (2017). Adversarial examples in the physical world. [ArXiv:1607.02533](http://arxiv.org/abs/1607.02533) [Cs, Stat].

[3] &emsp; Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. [ArXiv:1312.6199](http://arxiv.org/abs/1312.6199) [Cs].

[4] &emsp; Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. [ArXiv:1412.6572](http://arxiv.org/abs/1412.6572) [Cs, Stat].

[5] &emsp; Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., & Ma, A. (2019). Robustness May Be at Odds with Accuracy. 23.


[6] &emsp; Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., & Madry, A. (2019). Adversarial Examples Are Not Bugs, They Are Features. ArXiv:1905.02175 [Cs, Stat]. [http://arxiv.org/abs/1905.02175](http://arxiv.org/abs/1905.02175)

[7]&emsp;  Dubey, A., Cammarota, R., & Aysu, A. (2019). MaskedNet: The First Hardware Inference Engine Aiming Power Side-Channel Protection. ArXiv:1910.13063 [Cs]. [http://arxiv.org/abs/1910.13063](http://arxiv.org/abs/1910.13063])

[8] &emsp; Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016). DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [https://doi.org/10.1109/cvpr.2016.282](https://doi.org/10.1109/cvpr.2016.282)

[9] &emsp; Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2015). The Limitations of Deep Learning in Adversarial Settings. ArXiv:1511.07528 [Cs, Stat]. [http://arxiv.org/abs/1511.07528](http://arxiv.org/abs/1511.07528])

[10] &emsp; Su, J., Vargas, D. V., & Kouichi, S. (2019). One pixel attack for fooling deep neural networks. IEEE Transactions on Evolutionary Computation, 23(5), 828–841. [https://doi.org/10.1109/TEVC.2019.2890858](https://doi.org/10.1109/TEVC.2019.2890858)

[11] &emsp; Elsayed, G. F., Shankar, S., Cheung, B., Papernot, N., Kurakin, A., Goodfellow, I., & Sohl-Dickstein, J. (2018). Adversarial Examples that Fool both Computer Vision and Time-Limited Humans. ArXiv:1802.08195 [Cs, q-Bio, Stat]. [http://arxiv.org/abs/1802.08195](http://arxiv.org/abs/1802.08195)

[12] &emsp; Papernot, N., Faghri, F., Carlini, N., Goodfellow, I., Feinman, R., Kurakin, A., Xie, C., Sharma, Y., Brown, T., Roy, A., Matyasko, A., Behzadan, V., Hambardzumyan, K., Zhang, Z., Juang, Y.-L., Li, Z., Sheatsley, R., Garg, A., Uesato, J., … McDaniel, P. (2018). Technical Report on the CleverHans v2.1.0 Adversarial Examples Library. ArXiv:1610.00768 [Cs, Stat]. [http://arxiv.org/abs/1610.00768](http://arxiv.org/abs/1610.00768)