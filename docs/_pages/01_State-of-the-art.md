---

layout: page
title: Background
permalink: /state_of_the_art/
---

An adversarial example is a slightly modified input that is designed to mislead a machine learning model. In the context of this project we focus on images as inputs. However, adversarial examples exists for other domains as well (e.g. <a href="https://nicholas.carlini.com/code/audio_adversarial_examples">audio</a>) [[1]](#cite1). To craft such deceptive inputs, a clean input is often used and modified in a way that reduces the target neural network's confidence on the correct label. The induced perturbations are designed to be so subtle that they are hardly perceptible to a human. 

These manipulations can even occur in the physical world by modifying the appearance of an object [[2]](#cite2). With the adoption of neural networks in autonomous vehicles for example the existence of adversarial examples can cause serious safety concerns such as misreading road markings or stop signs. 

## Why adversarial examples exist
Since the discovery of adversarial examples [[3]](#cite3), varying explanations for this phenomenon have been given. The first hypothesis was that they were caused by the highly non-linear nature of neural networks, creating predictions for inputs wherefor there are no nearby training examples [[3]](#cite3). The authors also found that adversarial examples transfer between different models which is often referred to as called *transferability*.

In 2015, this hypothesis was overturned in favour of another explanation [[4]](#cite4) suggesting that the linear behaviour of networks (use of Rectified Linear Units, linear behaviour of sigmoids around 0, etc.) allow for the existence of adversarial examples [[4]](#cite4) and introduced the *Fast Gradient Sign Method* (FGSM) to generate them.

Since then multiple attack methods based on this assumption of the linear behaviour have been developed. We present some of them below.

Recently the focus in the search of the origin of adversarial examples has shifted away from the networks to the data. [[5, 6]](#cite5) claim that the existence of non-robust features in the data enables adversarial examples. These non-robust features are input features that help correctly classify an image in a standard setting, but will hinder accuracy under adversarial attacks. 

It is possible to train a robust classifier by using robust features only which are features that help correctly classify an image in both standard and adversarial settings.  The price for only using robust features to resist adversarial attacks is lower accuracy. Intuitively, this is because the classifier would rely on fewer features that could help correctly classify an image. Interestingly, Tsipras et al. [[6]](#cite6) show that these robust features tend to be features that humans can use to classify the image, making them more interpretable to humans and useful to GANs, which we cover in the next project. 

Tsipras et al. [[6]](#cite6) also show that a model trained only on adversarially-perturbed data with their respective incorrect labels can still yield good accuracy in the standard setting, supporting the claim that neural network models rely on imperceptible features. Additionally, in a second experiment they showed that using a robust dataset can transfer robustness to different networks. Their findings are discussed in this [article](https://distill.pub/2019/advex-bugs-discussion/).

## Existing attack methods and how to generate them
Existing attack methods can be grouped into the following categories.

**White box**: The attacker has full access to the model with all its parameters.

**Black box with probing**: An attacker has no access to the model's parameters. However, the model can be be queried to approximate the gradients.

**Black box without probing**: Here, the attacker has neither access nor can he query the model.

**Digital attack**: An attacker has direct access to digital data fed into the model.

Recently, [[7]](#cite7) have shown that network parameter can be extracted by analyzing the power consumption of the model during inference, a method called differential power analysis.

Moreover, attacks can be **targeted** or **untargeted**. In the latter scenario the attack is successful if any wrong class is predicted. For the former attack type, a specific class is targeted. Most existing attack method require a gradient to work with. That's why in the case of black box attacks it is common to approximate the gradient by taking advantage of transferability.

The following are the methods that we explore in this project.

- Fast Gradient Sign Method (FGSM) by [[4]](#cite4)
- Basic Iterative Method (BIM) by [[2]](#cite2)
- Iterative Least Likely Class Method (ILLM) by [[2]](#cite2)
- Deep Fool [[8]](#cite8)
- Targeted [Papernot et al.]


### Targeted Papernot et al
The truly targeted method by [[9]](#cite9) also does not modify all pixels and is distinct form the other methods in another way. Instead of generating adversarial examples from outputs the authors introduce a method which crafts an adversarial example from an input using the forward gradient of the network. This is achieved by iteratively computing saliency maps. From these, a specified number of features with the highest impact when perturbed are selected and adjusted according to a specified amount. This procedure stops when either the sample is classified as the desired target, the perturbation reaches a defined maximum or the maximum number of iterations is reached. This method does not modify all pixels. The results are highly effective, truly targeted as specified by the user and hardly perceptible for humans.


### Other methods
The methods above modify all pixels to increase the loss. [[10]](#cite10) propose that rather than modifying all pixels slightly, only one (or few) pixels can be modified with greater magnitude. The result is an overall less perturbed image. The modification is generated by differential evolution.

In addition to the perceptibility of the perturbations it has been shown that adversarial examples can even fool humans when only briefly exposed to the image [[11]](#cite11). This is significant since it would suggest transferability to the human brain.


## How to defend against them
As of this writing the best defence against adversarial examples is to onclude them in the training data of the model (adversarial training). [[12]](#cite12) created a library to support this by providing the common attack methods.


__________________

<a name="cite1"></a>
## References

<a name="cite2"></a>
[1] &emsp; Qin, Y., Carlini, N., Goodfellow, I., Cottrell, G., & Raffel, C. (n.d.). Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition. 12.

<a name="cite3"></a>
[2] &emsp; Kurakin, A., Goodfellow, I., & Bengio, S. (2017). Adversarial examples in the physical world. [ArXiv:1607.02533](http://arxiv.org/abs/1607.02533) [Cs, Stat].

<a name="cite4"></a>
[3] &emsp; Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. [ArXiv:1312.6199](http://arxiv.org/abs/1312.6199) [Cs].

<a name="cite5"></a>
[4] &emsp; Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. [ArXiv:1412.6572](http://arxiv.org/abs/1412.6572) [Cs, Stat].

<a name="cite6"></a>
[5] &emsp; Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., & Ma, A. (2019). Robustness May Be at Odds with Accuracy. 23.

<a name="cite7"></a>
[6] &emsp; Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., & Madry, A. (2019). Adversarial Examples Are Not Bugs, They Are Features. ArXiv:1905.02175 [Cs, Stat]. [http://arxiv.org/abs/1905.02175](http://arxiv.org/abs/1905.02175)

<a name="cite8"></a>
[7]&emsp;  Dubey, A., Cammarota, R., & Aysu, A. (2019). MaskedNet: The First Hardware Inference Engine Aiming Power Side-Channel Protection. ArXiv:1910.13063 [Cs]. [http://arxiv.org/abs/1910.13063](http://arxiv.org/abs/1910.13063])

<a name="cite9"></a>
[8] &emsp; Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016). DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [https://doi.org/10.1109/cvpr.2016.282](https://doi.org/10.1109/cvpr.2016.282)

<a name="cite10"></a>
[9] &emsp; Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2015). The Limitations of Deep Learning in Adversarial Settings. ArXiv:1511.07528 [Cs, Stat]. [http://arxiv.org/abs/1511.07528](http://arxiv.org/abs/1511.07528])

<a name="cite11"></a>
[10] &emsp; Su, J., Vargas, D. V., & Kouichi, S. (2019). One pixel attack for fooling deep neural networks. IEEE Transactions on Evolutionary Computation, 23(5), 828–841. [https://doi.org/10.1109/TEVC.2019.2890858](https://doi.org/10.1109/TEVC.2019.2890858)

<a name="cite12"></a>
[11] &emsp; Elsayed, G. F., Shankar, S., Cheung, B., Papernot, N., Kurakin, A., Goodfellow, I., & Sohl-Dickstein, J. (2018). Adversarial Examples that Fool both Computer Vision and Time-Limited Humans. ArXiv:1802.08195 [Cs, q-Bio, Stat]. [http://arxiv.org/abs/1802.08195](http://arxiv.org/abs/1802.08195)

<a name="cite13"></a>
[12] &emsp; Papernot, N., Faghri, F., Carlini, N., Goodfellow, I., Feinman, R., Kurakin, A., Xie, C., Sharma, Y., Brown, T., Roy, A., Matyasko, A., Behzadan, V., Hambardzumyan, K., Zhang, Z., Juang, Y.-L., Li, Z., Sheatsley, R., Garg, A., Uesato, J., … McDaniel, P. (2018). Technical Report on the CleverHans v2.1.0 Adversarial Examples Library. ArXiv:1610.00768 [Cs, Stat]. [http://arxiv.org/abs/1610.00768](http://arxiv.org/abs/1610.00768)