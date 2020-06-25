---

layout: page
title: Results
permalink: /results/
---


In this section we discuss the performance of the different attacks. First we briefly look at the properties of the clean data. Then we attack the model with different models and analyze the success.

## Data Exploration
The goal of adversarial examples is to fool the network into predicting a wrong class with high confidence. That's why we inspect the confidence of the network on the clean data first.

In the data there are 452 out of 1000 distinct classes represented. The most frequent class is `ballplayer, baseball player` (class index 981) with 8 occurrences, followed by `racer, race car, racing car`, `stone wall` and `worm fence, snake fence, snake-rail fence, ...` with 7 each. Within these frequent classes the model's confidence is around $$58$$% with a standard deviation of around $$27$$. This is probably due to false predictions.

The average confidence for a class is between $$99.99$$% (`barrel, cask`) and $$41.64$$% (`sandal`). In the plot the distribution of the average confidence can be seen. For over half of the examples the model has an average confidence of over $$60$$% and over $$\frac{2}{3}$$ of the examples have a confidence of over $$50$$%.


![name of the image](/Adversarial_Examples_GANs/assets/Data_Exploration-Average-confidence_per_class.png){:class="img-responsive"}{:height="100%" width="100%"}


The table shows the model's overall performance.


|                | Confidence    |  Accuracy     |
| :------------- | :----------:  | :----------:  | 
|  Top 1         | 0.69          | 0.84          |
|  Top 5         | 0.63          | 0.97          |

Top 1 means that the predicted class is the correct class. Top 5 means that the correct class is among the 5 predicted classes with the highest score.


## Fast Gradient Sign Method

The following is an example of the original image, the generated perturbance and the resulting adversarial image using the Fast Gradient Sign Method:

![FGSM Attack](/Adversarial_Examples_GANs/assets/Sample_766_pair.png){:class="img-responsive"}{:height="100%" width="100%"}

The adversarial image appears slightly more blurry than the original one, like for example taken at poor a resolution. Without the reference image however, it can be hard to tell that it has been modified. With increasing attack strength this becomes more and more obvious as can be seen in the following images:

![FGSM Attack](/Adversarial_Examples_GANs/assets/Sample_766_series.png){:class="img-responsive"}{:height="100%" width="100%"}

Here the values for epsilon are: 0, 4/255, 8/255, 12/255 and 16/255. The image appears more and more noisy. We later show methods which produce cleaner looking adversaries.

But how effective is this attack in tricking the network? Recall that we want to produce images which cause a prediction of a wrong class at a high confidence. From the figure below you can see that the confidence drops sharply while the class changes only for a few epsilons. The example above is the sample with which the model has the highest confidence on clean data. We have seen this behaviour with other images with high confidence in the dataset. This leads us to the first hypothesis:

**Hypothesis 1:** Images with a high initial confidence are harder to manipulate.


### All Images
We consider a plot of accuracy and confidence over the attack strength epsilon. For this hypothesis to be true we would observe a sharp drop in accuracy with increasing attack strength. The higher the initial confidence is, the smaller the slope of the accuracy should be. It is harder to attack the network at the same epsilon. At the same time the confidence should drop slightly since more and more robust features are altered.

Recall from the section Data Exploration how the confidence over all data is distributed. We consider correct initial classifications only and split the data by confidences in the interval of 5% points.

![FGSM Attack](/Adversarial_Examples_GANs/assets/Accuracies_Confidences.png){:class="img-responsive"}{:height="100%" width="100%"}

## BIM

