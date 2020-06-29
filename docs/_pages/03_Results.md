---

layout: page
title: Results
permalink: /results/
---

In this section we discuss the performance of the different attacks. First we briefly look at the properties of the clean data. Then we attack the model with different models and analyze the success.

## Data Exploration
The goal of adversarial examples is to fool the network into predicting a wrong class with high confidence. That's why we inspect the confidence of the network on the clean data first.

In the data there are 452 out of 1000 distinct classes represented. The most frequent class is `ballplayer, baseball player` (class index 981) with 8 occurrences, followed by `racer, race car, racing car`, `stone wall` and `worm fence, snake fence, snake-rail fence, ...` with 7 each. Within these frequent classes the model's confidence is around $$58$$% with a standard deviation of around $$27$$. This is probably due to false predictions. Figure 1 shows how the the top 5 confidence is distributed for different initial confidences.

{% include image.html file="Confidence_Distributions.png" description="Figure 1: Top 5 confidences from 100% to 60% in 5% steps." %}

The average confidence for a class is between $$99.99$$% (`barrel, cask`) and $$41.64$$% (`sandal`). The distribution of the average confidence is shown in figure 2. For over half of the examples the model has an average confidence of over $$60$$% and over $$\frac{2}{3}$$ of the examples have a confidence of over $$50$$%.

{% include image.html file="Data_Exploration-Average-confidence_per_class.png" description="Figure 2: Average confidence per class in descending order. The majority of classes consists of one sample only. The highest number of samples per class is 8." %}


The table shows the model's overall performance.


|                | Confidence    |  Accuracy     |
| :------------- | :----------:  | :----------:  | 
|  Top 1         | 0.69          | 0.84          |
|  Top 5         | 0.63          | 0.97          |

Top 1 means that the predicted class is the correct class. Top 5 means that the correct class is among the 5 predicted classes with the highest score.


## Fast Gradient Sign Method

The following is an example of the original image, the generated perturbance and the resulting adversarial image using the Fast Gradient Sign Method:

{% include image.html file="Sample_766_pair.png" description="Figure 3: Original image, the with the FGSM generated perturbance and resulting adversarial image. This attack decreases the network’s confidence from almost 100% down to 14%" %}

The adversarial image appears slightly more blurry than the original one, like for example taken at poor a resolution. Without the reference image however, it can be hard to tell that it has been modified. With increasing attack strength this becomes more and more obvious as can be seen in the following images:

{% include image.html file="Sample_766_series.png" description="Figure 4: Original image and a series of adversarial images with increasing attack intensity. The values for epsilon are: 0, 4/255, 8/255, 12/255 and 16/255. Subjectively, at around 12/255 the attack becomes noticeable." %}

The image appears more and more noisy. We later show methods which produce cleaner looking adversaries.

But how effective is this attack in tricking the network? Recall that we want to produce images which cause a prediction of a wrong class at a high confidence. From the figure below you can see that the confidence drops sharply while the class changes only for a few epsilons. The example above is the sample with which the model has the highest confidence on clean data. We have seen this behaviour with other images with high confidence in the dataset. This leads us to the first hypothesis:

**Hypothesis 1:** Images with a high initial confidence are harder to manipulate.


### All Images
We consider a plot of accuracy and confidence over the attack strength epsilon. For this hypothesis to be true we would observe a sharp drop in accuracy with increasing attack strength. The higher the initial confidence is, the smaller the slope of the accuracy should be. It is harder to attack the network at the same epsilon. At the same time the confidence should drop slightly since more and more robust features are altered.

Recall from the section Data Exploration how the confidence over all data is distributed. We consider correct initial classifications only and split the data by confidences in the interval of 5% points.

{% include image.html file="Accuracies_Confidences.png" description="Figure 5: Accuracy (left) and confidence (right) for different initial confidences over increasing attack intensity. Only if the prediction on the clean image is correct an adversary is generated." %}


### Individual Images
We find some exceptions. Figure 6 shows two samples where the FGSM is not able to change the class. Moreover, also the lowest confidence remains fairly large. After an initial sudden decrease in confidence, it increases again. In the first example even almost up to clean levels.

{% include image.html file="Individual_Images-Same_Class.png" description="Figure 6: Two examples for class-invariance under FGSM. On the left is the clean example. Next to it the confidence and if class is correct or not over multiple epsilons. The third plot shows the top 5 confidence for the clean case, whereas the rightmost plot shows the top 5 confidence for the worst case. The confidence increases again after an initial dip." %}

A smaller increase in confidence after a dip can also be seen in figure 7. Despite the high initial confidence the adversary is able to change the class here. However, with a stronger attack the model predicts the correct class again. We found this behaviour of a bounce back to the correct class frequently for high initial confidences. Around the epsilon where the class changes we found that the highest and second highest confidences are very similar, whereas the gap to the third highest remains. The decrease in prediction confidence causes the second highest confidence to grow over proportionally which leads to the swap at the lowest point.

{% include image.html file="Individual_Images-Bounce_Back.png" description="Figure 7: Two examples where the adversary is able to change the class. However, for further increase in attack intensity the model recovers and predicts the correct class while being increasingly confident again. The rightmost plot shows the top 5 confidence for the smallest epsilon where a false class is predicted."%}

The previous examples showed images with high initial confidence. It is not possible to generate an adversary with higher confidence than the original one for these cases. Figure 8 shows where it is possible. The initial confidence is quite low (around 50%). The FGSM can manipulate this image easily and achieves a confidence of 80%!

{% include image.html file="Individual_Images-Sample_258.png" description="Figure 8:  Example for low initial confidence and greater adversarial confidence. Note how a small perturbation achieves the best results here. In this particular attack the perturbation is so slight that it is not representable in an 8bit image."%}

After analyzing the FGSM we now turn to a method derived from it, the basic iterative method BIM.

## BIM

