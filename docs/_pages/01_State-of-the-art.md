---

layout: page
title: Background
permalink: /state_of_the_art/
---

An adversarial example is a slightly modified input that is designed to mislead a machine learning model. In the context of this project we focus on images as inputs. However, adversarial examples exists for other domains as well (e.g. <a href="https://nicholas.carlini.com/code/audio_adversarial_examples">audio</a>) [1]. To craft such deceptive inputs, a clean input is often used and modified in a way that reduces the target neural network's confidence on the correct label. The induced perturbations are designed to be so subtle that they are hardly perceptible to a human. 

These manipulations can even occur in the physical world by modifying the appearance of an object [2]. With the adoption of neural networks in autonomous vehicles for example the existence of adversarial examples can cause serious safety concerns such as misreading road markings or stop signs. Since the discovery of their existence in 2013 [3] research has focussed on understanding their origins and how to defend against them.


## Why adversarial examples exist
Since the discovery of adversarial examples [3], varying explanations for this phenomenon have been given. The first hypothesis was that they were caused by the highly non-linear nature of neural networks [3]. The authors also found that adversarial examples transfer between different models called *transferability*.

In 2015, this hypothesis was rejected. Instead, [4] suggest that the linear behaviour of networks (use of Rectified Linear Units, linear behaviour of sigmoids around 0, etc.) enables adversarial examples and introduced the *Fast Gradient Sign Method* (FGSM) to generate them.

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

- Fast Gradient Sign Method (FGSM) by [4]
- Basic Iterative Method (BIM) by [2]
- Iterative Least Likely Class Method (ILLM) by [2]
- Deep Fool [8]
- Targeted [Papernot et al.]


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