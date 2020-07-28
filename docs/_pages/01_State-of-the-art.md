---

layout: page
title: Background
permalink: /state_of_the_art/
---

An adversarial example is a slightly modified input that is designed to mislead a machine learning model. In the context of this project we focus on images as inputs and deep neural nets as models. However, adversarial examples exists for other domains as well (e.g. <a href="https://nicholas.carlini.com/code/audio_adversarial_examples">audio</a>) [[1]](#cite1). To craft such deceptive inputs, a clean input is often used and modified in a way that reduces the target neural network's confidence on the correct label. The induced perturbations are designed to be so subtle that they are hardly perceptible to a human.

These manipulations can even occur in the physical world by modifying the appearance of an object [[2]](#cite2). With the adoption of neural networks in autonomous vehicles for example the existence of adversarial examples can cause serious safety concerns such as misreading road markings or stop signs [[3]](#cite3).

Deep neural networks can be attacked in both the training or inference phase [[4]](#cite4). We focus on the latter.

## Why adversarial examples exist
Since the discovery of adversarial examples targeting neural network classifiers [[5]](#cite5), varying explanations for this phenomenon have been given. The first hypothesis was that they were caused by the highly non-linear nature of neural networks, creating predictions for inputs wherefor there are no nearby training examples [[5]](#cite5). The authors also found that adversarial examples transfer between different models which is often referred to as called *transferability*.

In 2015, this hypothesis was overturned in favour of another explanation [[6]](#cite6) suggesting that the linear behaviour of networks (use of Rectified Linear Units, linear behaviour of sigmoids around 0, etc.) allow for the existence of adversarial examples [[6]](#cite6) and introduced the *Fast Gradient Sign Method* (FGSM) to generate them.

Since then multiple attack methods based on this assumption of the linear behaviour have been developed. We present some of them below.

Recently the focus in the search of the origin of adversarial examples has shifted away from the networks to the data. [[7, 8]](#cite7) claim that the existence of non-robust features in the data enables adversarial examples. These non-robust features are input features that help correctly classify an image in a standard setting, but will hinder accuracy under adversarial attacks. 

It is possible to train a robust classifier by using robust features only which are features that help correctly classify an image in both standard and adversarial settings.  The price for only using robust features to resist adversarial attacks is lower accuracy. Intuitively, this is because the classifier would rely on fewer features that could help correctly classify an image. Interestingly, Tsipras et al. [[8]](#cite8) show that these robust features tend to be features that humans can use to classify the image, making them more interpretable to humans and useful to GANs, which we cover in the next project. 

Tsipras et al. [[8]](#cite8) also show that a model trained only on adversarially-perturbed data with their respective incorrect labels can still yield good accuracy in the standard setting, supporting the claim that neural network models rely on imperceptible features. Additionally, in a second experiment they showed that using a robust dataset can transfer robustness to different networks. Their findings are discussed in this [article](https://distill.pub/2019/advex-bugs-discussion/).

## Existing attack methods and how to generate them
Existing attack methods can be grouped into the following categories.

**White box**: The attacker has full access to the model with all its parameters.

**Black box with probing**: An attacker has no access to the model's parameters. However, the model can be be queried to approximate the gradients.

**Black box without probing**: Here, the attacker has neither access nor can he query the model.

**Digital attack**: An attacker has direct access to digital data fed into the model.

Moreover, attacks can be **targeted** or **untargeted**. In the latter scenario the attack is successful if any wrong class is predicted. For the former attack type, a specific class is targeted. Most existing attack method require a gradient to work with. Consequently, most black box attacks take advantage of transferability. Papernot et al. have shown that adversarial examples also transfer between different machine learning models and training datasets [[9,10]](#cite9). Recently, [[11]](#cite10) have shown that network parameter can also be extracted by analyzing the power consumption of the model during inference.

The following are the methods that we explore in this project.

- Fast Gradient Sign Method (FGSM) by [[6]](#cite6)
- Basic Iterative Method (BIM) by [[3]](#cite3)
- Iterative Least Likely Class Method (ILLM) by [[3]](#cite3)
- DeepFool [[12]](#cite12)

### Other methods
The methods above modify all pixels to increase the loss. [[13]](#cite13) propose that rather than modifying all pixels slightly, only one (or few) pixels can be modified with greater magnitude. The result is an overall less perturbed image. The modification is generated by differential evolution. [[14]](#cite14) introduce a method which does not rely on backpropagation to craft adversaries.

The strongest attacks to date have been proposed by Carlini and Wagner [[15]](#cite15). Their attacks are based on three distance metrics which assure that the adversaries are strong, imperceptible and achieve a high confidence. In contrast to BIM or DeepFool, these attacks can target any desired class.

## How to defend against them
As of this writing the best defense against adversarial examples is to include them in the training data of the model (adversarial training [[6]](#cite6)). This allows learning robust decision boundaries, which works better for models with large capacity [[16]](#cite16). Papernot et al. [[17]](#cite17) created the Cleverhans library to support adversarial training by providing the common attack methods.

A second approach is using another model which is specialized in detecting adversaries [[18]](#cite18). A third approach is defensive distillation [[19]](#cite19). However, Carlini et al. have shown that this cannot defend against their strong attacks [[16]](#cite16).

## Beyond classification
Adversarial examples also exist for semantic segmentation, object detection or pose estimation tasks. Two common algorithms to generate them are *Dense Adversary Generation* [[20]](#cite20) and *Houdini* [[21]](#cite21). Xiao et al. [[22]](#cite22) analyze these and find that for semantic segmentation adversaries do not transfer between models. Moreover, with spatial consistency check they introduce a promising detection mechanism for the segmentation task.

__________________

# References

<a name="cite1"></a>
[1] &emsp; Qin, Y., Carlini, N., Goodfellow, I., Cottrell, G., & Raffel, C. (n.d.). Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition. 12.

<a name="cite2"></a>
[2] &emsp; Kurakin, A., Goodfellow, I., & Bengio, S. (2017). Adversarial examples in the physical world. [ArXiv:1607.02533](http://arxiv.org/abs/1607.02533) [Cs, Stat].

<a name="cite3"></a>
[3] &emsp; Eykholt, K., Evtimov, I., Fernandes, E., Li, B., Rahmati, A., Xiao, C., Prakash, A., Kohno, T., & Song, D. (2018). Robust Physical-World Attacks on Deep Learning Models. ArXiv:1707.08945 [Cs]. [http://arxiv.org/abs/1707.08945](http://arxiv.org/abs/1707.08945)

<a name="cite4"></a>
[4] &emsp; Huang, L., Joseph, A. D., Nelson, B., Rubinstein, B. I. P., & Tygar, J. D. (n.d.). Adversarial Machine Learning. 15.

<a name="cite5"></a>
[5] &emsp; Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. [ArXiv:1312.6199](http://arxiv.org/abs/1312.6199) [Cs].

<a name="cite6"></a>
[6] &emsp; Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. [ArXiv:1412.6572](http://arxiv.org/abs/1412.6572) [Cs, Stat].

<a name="cite7"></a>
[7] &emsp; Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., & Ma, A. (2019). Robustness May Be at Odds with Accuracy. 23.

<a name="cite8"></a>
[8] &emsp; Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., & Madry, A. (2019). Adversarial Examples Are Not Bugs, They Are Features. ArXiv:1905.02175 [Cs, Stat]. [http://arxiv.org/abs/1905.02175](http://arxiv.org/abs/1905.02175)

<a name="cite9"></a>
[9] &emsp; Papernot, N., McDaniel, P., & Goodfellow, I. (2016). Transferability in Machine Learning: From Phenomena to Black-Box Attacks using Adversarial Samples. ArXiv:1605.07277 [Cs]. [http://arxiv.org/abs/1605.07277](http://arxiv.org/abs/1605.07277)

<a name="cite10"></a>
[10] &emsp; Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2017). Practical Black-Box Attacks against Machine Learning. ArXiv:1602.02697 [Cs]. [http://arxiv.org/abs/1602.02697](http://arxiv.org/abs/1602.02697)

<a name="cite11"></a>
[11]&emsp;  Dubey, A., Cammarota, R., & Aysu, A. (2019). MaskedNet: The First Hardware Inference Engine Aiming Power Side-Channel Protection. ArXiv:1910.13063 [Cs]. [http://arxiv.org/abs/1910.13063](http://arxiv.org/abs/1910.13063])

<a name="cite12"></a>
[12] &emsp; Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016). DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [https://doi.org/10.1109/cvpr.2016.282](https://doi.org/10.1109/cvpr.2016.282)

<a name="cite13"></a>
[13] &emsp; Su, J., Vargas, D. V., & Kouichi, S. (2019). One pixel attack for fooling deep neural networks. IEEE Transactions on Evolutionary Computation, 23(5), 828–841. [https://doi.org/10.1109/TEVC.2019.2890858](https://doi.org/10.1109/TEVC.2019.2890858)

<a name="cite14"></a>
[14] &emsp; Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2015). The Limitations of Deep Learning in Adversarial Settings. ArXiv:1511.07528 [Cs, Stat]. [http://arxiv.org/abs/1511.07528](http://arxiv.org/abs/1511.07528])

<a name="cite15"></a>
[15] &emsp; Carlini, N., & Wagner, D. (2017). Towards Evaluating the Robustness of Neural Networks. ArXiv:1608.04644 [Cs]. [http://arxiv.org/abs/1608.04644](http://arxiv.org/abs/1608.04644)

<a name="cite16"></a>
[16] &emsp; Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards Deep Learning Models Resistant to Adversarial Attacks. ArXiv:1706.06083 [Cs, Stat]. [http://arxiv.org/abs/1706.06083](http://arxiv.org/abs/1706.06083)

<a name="cite17"></a>
[17] &emsp; Papernot, N., Faghri, F., Carlini, N., Goodfellow, I., Feinman, R., Kurakin, A., Xie, C., Sharma, Y., Brown, T., Roy, A., Matyasko, A., Behzadan, V., Hambardzumyan, K., Zhang, Z., Juang, Y.-L., Li, Z., Sheatsley, R., Garg, A., Uesato, J., … McDaniel, P. (2018). Technical Report on the CleverHans v2.1.0 Adversarial Examples Library. ArXiv:1610.00768 [Cs, Stat]. [http://arxiv.org/abs/1610.00768](http://arxiv.org/abs/1610.00768)

<a name="cite18"></a>
[18] &emsp; Lu, J., Issaranon, T., & Forsyth, D. (2017). SafetyNet: Detecting and Rejecting Adversarial Examples Robustly. ArXiv:1704.00103 [Cs]. [http://arxiv.org/abs/1704.00103](http://arxiv.org/abs/1704.00103)

<a name="cite19"></a>
[19] &emsp; Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. ArXiv:1503.02531 [Cs, Stat]. [http://arxiv.org/abs/1503.02531](http://arxiv.org/abs/1503.02531)

<a name="cite20"></a>
[20] &emsp; Xie, C., Wang, J., Zhang, Z., Zhou, Y., Xie, L., & Yuille, A. (2017). Adversarial Examples for Semantic Segmentation and Object Detection. ArXiv:1703.08603 [Cs]. [http://arxiv.org/abs/1703.08603](http://arxiv.org/abs/1703.08603)

<a name="cite21"></a>
[21] &emsp; Cisse, M., Adi, Y., Neverova, N., & Keshet, J. (2017). Houdini: Fooling Deep Structured Prediction Models. ArXiv:1707.05373 [Cs, Stat]. [http://arxiv.org/abs/1707.05373](http://arxiv.org/abs/1707.05373)

<a name="cite22"></a>
[22] &emsp; Xiao, C., Deng, R., Li, B., Yu, F., Liu, M., & Song, D. (2018). Characterizing Adversarial Examples Based on Spatial Consistency Information for Semantic Segmentation. ArXiv:1810.05162 [Cs]. [http://arxiv.org/abs/1810.05162](http://arxiv.org/abs/1810.05162)
