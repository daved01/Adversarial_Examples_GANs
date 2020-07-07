---

layout: page
title: Results
permalink: /results/
---

In this section we discuss the performance of the different attacks. First we briefly look at the properties of the clean data. Then we attack the model with different models and analyze the success.

## 0. Data Exploration
Considering that the goal of adversarial examples is to fool the network into predicting a wrong class with high confidence, a good place to start start is to inspect the model's performance without any adversarial perturbation.

In the data there are 452 out of 1000 distinct classes represented. The most frequent class is `ballplayer, baseball player` (class index 981) with 8 occurrences, followed by `racer, race car, racing car`, `stone wall` and `worm fence, snake fence, snake-rail fence, ...` with 7 each. Within these frequent classes, the model's confidence is around $$58$$% with a standard deviation of around $$27$$. This large range in confidence is likely due to false predictions. Figure 1 shows how the the top 5 confidence is distributed for different confidences in their top predictions.

{% include image.html file="Confidence_Distributions.png" description="Figure 1: Top 5 confidences from 100% to 60% in 5% steps." %}

The average confidence for a class is between $$99.99$$% (`barrel, cask`) and $$41.64$$% (`sandal`). The distribution of the average confidence is shown in figure 2. For over half of the examples, the model has an average confidence of over $$60$$% and over $$\frac{2}{3}$$ of the examples have a confidence of over $$50$$%.

{% include image.html file="Data_Exploration-Average-confidence_per_class.png" description="Figure 2: Average confidence per class in descending order. The majority of classes consist of only one sample. The highest number of samples per class is 8." %}


The table shows the model's overall performance.


|                | Confidence    |  Accuracy     |
| :------------- | :----------:  | :----------:  | 
|  Top 1         | 0.69          | 0.84          |
|  Top 5         | 0.63          | 0.97          |

Top 1 means that the predicted class is the correct class. Top 5 means that the correct class is among the 5 predicted classes with the highest score.


## 1. Fast Gradient Sign Method

The following is an example of the original image, the generated perturbance, and the resulting adversarial image using the Fast Gradient Sign Method:

{% include image.html file="Sample_766_pair.png" description="Figure 3: Original image, the with the FGSM generated perturbance and resulting adversarial image. This attack decreases the network’s confidence from almost 100% down to 14%." %}

The adversarial image appears slightly blurrier than the original one, like for example taken at poorer resolution or with a worse camera. Without the reference image however, it can be difficult to tell that it has been modified. With increasing attack strength, this becomes noticeable as can be seen in the following images:

{% include image.html file="Sample_766_series.png" description="Figure 4: Original image and a series of adversarial images with increasing attack intensity. The values for epsilon are: 0, 4/255, 8/255, 12/255 and 16/255. Subjectively, at around 12/255 the attack becomes noticeable." %}

The image appears more and more noisy. We later show methods which produce cleaner looking adversaries.

But how effective is this attack in tricking the network? Recall that we want to produce images which cause a prediction of a wrong class at a high confidence. From figure 7 you can see that the confidence drops sharply while the class changes only for a few epsilons. The example above is the sample with which the model has one of the highest confidences on clean data. We have seen this same behaviour with other images with high confidence in the dataset. This leads us to the first hypothesis:

**Hypothesis 1:** Images with a high initial confidence are harder to manipulate.


### 1.1 All Images
We consider a plot of accuracy and confidence over the attack strength epsilon. For this hypothesis to be true we would observe a sharp drop in accuracy with increasing attack strength. The higher the initial confidence is, the smaller the slope of the accuracy should be. It is harder to attack the network at the same epsilon. At the same time the confidence should drop slightly since more and more robust features are altered.

Recall from the section Data Exploration how the confidence over all data is distributed. We consider correct initial classifications only and split the data by confidences in ranges of 5% points.

{% include image.html file="Accuracies_Confidences.png" description="Figure 5: Accuracy (left) and confidence (right) for different initial confidences over increasing attack intensity. Examples that are labelled incorrectly by the model without adversarial perturbations are excluded." %}


### 1.2 Individual Images
In theory, increasing the size of the perturbation should lower the confidence of the model in predicting the correct class. We, however, have found some exceptions. Figure 6 shows two samples where the FGSM is not able to change the class while the model maintains a relatively high confidence. Interestingly, the confidence in the original correct class first quickly drops then increases again with increasing adversarial perturbance. In the first example, the model's confidence even reaches back to clean levels at the highest levels of perturbance.

{% include image.html file="Individual_Images-Same_Class.png" description="Figure 6: Two examples for class-invariance under FGSM. On the left is the clean example. Next to it the confidence and if class is correct or not over multiple epsilons. The third plot shows the top 5 confidence for the clean case, whereas the rightmost plot shows the top 5 confidence when the top class is at its lowest confidence. The confidence increases again after an initial dip." %}

A smaller increase in confidence after a dip can also be seen in figure 7. Despite the high initial confidence the adversary is able to change the class here. However, with a stronger attack the model predicts the correct class again. We found this behaviour of a bounce back to the correct class frequently for high initial confidences. Around the epsilon where the class changes we found that the highest and second highest confidences are very similar, whereas the gap to the third highest remains. The decrease in prediction confidence causes the second highest confidence to grow over proportionally which leads to the swap at the lowest point.

{% include image.html file="Individual_Images-Bounce_Back.png" description="Figure 7: Two examples where the adversary is able to change the class. However, for further increase in attack intensity the model recovers and predicts the correct class while being increasingly confident again. The rightmost plot shows the top 5 confidence for the smallest epsilon where a false class is predicted."%}

The previous examples showed images with high initial confidence. It is not possible to generate an adversary with higher confidence than the original one for these cases. Figure 8 shows where it is possible. The initial confidence is quite low (around 50%). The FGSM can manipulate this image easily and achieves a confidence of 80%!

{% include image.html file="Individual_Images-Sample_258.png" description="Figure 8: Example for low initial confidence and greater adversarial confidence. Note how a small perturbation achieves the best results here. In this particular attack the perturbation is so slight that it is not representable in an 8bit image."%}

After analyzing the FGSM we now turn to a method derived from it, the basic iterative method BIM.

## 2. BIM
The authors in [Adversarial Examples in the Physical World](http://arxiv.org/abs/1607.02533) introduce BIM as an extension of FGSM to generate stronger adversaries for higher computational costs. In figure 9 we show the influence of the two hyperparameters $$\alpha$$ and *num_iter* on the attack with the top image from figure 6. We see that BIM is able to generate an adversary which fools the network while FGSM cannot.


{% include image.html file="BIM-Hyperparameter_variation_132.png" description="Figure 9: Effects of the two hyperparameters for BIM. In each row the number of iterations increases from left to right at a constant alpha. We chose the alphas 1/255, 10/255, 68/255 and for number of iterations 1, 10, 15, 24. On the top left both alpha and number of iterations are 1. This attack is similar to FGSM with an additional clipping of each pixel value. Note how the number of iterations has a strong impact on the predicted class whereas alpha does not."%}

We can see that for this sample increasing alpha generally causes the confidence to drop for smaller epsilons, as shown on the vertical axis from the top left corner. By increasing alpha only it is not possible to change the predicted class. This is expected since alpha is the parameter from the “fast” part of BIM. However, changing the class is possible by increasing the number of iterations. The best results are in the top right plot (lowest alpha, highest number of iterations). For two epsilons the networks predicts a false class with almost 80% confidence. Considering that FGSM was not able to change the class at all this is a strong result. The authors recommend keeping alpha at 1/255 and changing the number of iterations with a heuristic based on epsilon. 

For the rest of this report we choose the hyperparameters as suggested by the authors. In figure 10 we attack with the same images as in figure 8.

{% include image.html file="BIM_Individual_Images-Dont_Bounce_Back.png" description="Figure 10: Example with high (top) and low (bottom) clean confidence. Clean image is on the left, the adversarial confidence for select epsilon second, third top 5 confidence for the clean case and top 5 confidence for the highest adversarial confidence. In contrast to FGSM BIM is able to generate six adversaries in the top case. For the bottom BIM is not able to achieve higher confidence than FGSM."%}

For the top the network is very confident in the clean case. FGSM is able to change the class at a low confidence for a few epsilon only. In contrast, BIM is able to achieve an adversarial confidence of almost 80%. Additionally, the “bounce-back” effect seen for FGSM is not present. Interestingly, for the bottom image BIM is not able to generate higher confidence with the adversary. Given that in both cases the best adversary is for small epsilon it is likely that for both methods the perturbations are imperceptible. We investigate the perceptibility of adversaries generated by different methods below. Note that this example falsifies hypothesis 1 for BIM since, measured by adversarial confidence, it appears to be harder to manipulate this image with low initial confidence than the example at the top.

So far it seems as if BIM is more successful in attacking the network. Let’s see how the general behaviour on all samples is.


### 2.1 All Images
As for FGSM, to test hypothesis 1 we generate adversaries for different initial confidence ranges. Figure 11 shows that BIM is able to generate adversaries much more consistently than FGSM. For an epsilon greater 10 all attacks are successful. The slopes for lower confidence ranges is like for FGSM greater.

{% include image.html file="BIM-Accuracies_Confidences.png" description="Figure 11: Top 1 accuracy and adversarial confidence for BIM and how they compare to FGSM. Overall BIM is able to generate adversaries more successfully and at higher confidences. Note that as for FGSM if the clean confidence is higher, it is on average harder to generate adversaries for smaller epsilons, which also have a lower confidence."%}

On the right side you can see that BIM generates not only more consistently, but also stronger adversaries. Interesting is the dip for low epsilons. For small epsilons BIM and FGSM are fairly similar, since we use calculate the number of iterations based on epsilon and keep alpha pixel-wise at 1. The more epsilon growth, the more different BIM becomes.

The two methods analyzed so far generate untargeted attacks. Next, we investigate the ILLM which targets the least likeliest class.


## 3. Least Likeliest Class Method
This method is related to BIM but is targeted by making the network predict the class with the lowest clean confidence. Recall from figure 1 how the confidences for each class look like. For a well-trained classifier there is a significant gap between the highest and second highest confidence. While BIM tries to increase the loss on the correct class, which then leads to the second class becoming more likely, ILLM tries to decrease the loss on the class with the lowest confidence. The result often is a low overall confidence as seen in figure 12. Note that ILLM is on average also less successful in attacking the model than the FGSM.

{% include image.html file="ILLM-compare_attacks_FGSM_BIM.png" description="Figure 12: Top 1 accuracy (left) and adversarial confidences (right) for FGSM, BIM and ILLM. The high probability that the adversarial class will be very different from the correct one comes at the cost of significantly worse performance in comparison to the related BIM method."%}


### 3.1 Individual Images

We find examples where ILLM performance is poor. In the top image of figure 13 the adversary tricks the network into believing it is looking at a hoop skirt rather than an apron. Considering that there are 1000 possible classes, including for example “baseball”, this choice seems poor since subjectively it is not very different. Additionally, the confidence is low. An adversary generated by BIM for example predicts “Arabian Camel” with 28% confidence.

{% include image.html file="ILLM-confidences_two_examples.png" description="Figure 13: Confidences for two ILLM attacks. In the top example ILLM is able to generate an adversary with low confidence only. The rightmost plot shows how flat the confidence distribution for the lowest successful attack is. In the bottom example the adversary generates the same class as BIM, however at a lower confidence."%}

The confidence distribution in the top image is as expected relatively flat. It appears as if ILLM gives much more confidence to each class rather than concentrating the confidence on a few classes, as explained above. In the bottom image of figure 13 you can see another example. With the first epsilon the adversary is able to change the class to “lionfish”, the same as BIM does. However, as you can see on the rightmost plot, the confidence iIma achieves over 80% adversarial confidence into same class. At the same time for ILLM the second class is higher than for BIM. This again supports the hypothesis that ILLM “flattens out” the confidence distribution rather than increasing the gap between 1st and 2nd.


### 3.2 Perceptibility
Finally, we look at perceptibility of the aforementioned attacks. In figure 14 you can see the image on the left and attacks with FGSM, BIM and ILLM. To make them comparable we attack with FGSM first and choose the highest adversarial confidence, which is around 24% here. Next we attack with BIM and find the first adversary for which the model has a similar confidence, also 24%. We proceed with ILLM.

{% include image.html file="ILLM-comparison_all_methods.png" description="Figure 14: Attacks with FGSM (second from left), BIM (third) and ILLM (right). FGSM and BIM achieve a similar adversarial confidence of around 24%. Since for ILLM the confidence is significantly lower, the highest adversarial confidence of 3% was chosen. Epsilons are 20, 4, 12 for FGSM, BIM and ILLM."%}

Subjectively, BIM generates the least perceptible adversary. At the same time it also generates the most confident adversary with around 60%. We have seen this for other images as well. BIM was for example able to change a correct, 90% confident detection of a fly to a 100% confident prediction of a bow (sample 894). ILLM on the other hand produces adversaries which cause predictions at overall low confidences.



## 4. Conclusions