---

layout: page
title: Data exploration
permalink: /data_exploration/
---

The goal of adversarial examples is to fool the network into predicting a wrong class with high confidence. That's why we inspect the confidence of the network on the clean data first.

In the data there are 452 out of 1000 distinct classes represented. The most frequent class is `ballplayer, baseball player` (class index 981) with 8 occurrences, followed by `racer, race car, racing car`, `stone wall` and `worm fence, snake fence, snake-rail fence, ...` with 7 each. Within these frequent classes the model's confidence is around $$58$$% with a standard deviation of around $$27$$. This is probably due to false predictions.

The average confidence for a class is between $$99.99$$% (`barrel, cask`) and $$41.64$$% (`sandal`). In the plot the distribution of the average confidence can be seen. For over half of the examples the model has an average confidence of over $$60$$% and over $$\frac{2}{3}$$ of the examples have a confidence of over $$50$$%.


![name of the image](/Adversarial_Examples_GANs/assets/Adversarial-Examples_Average-confidence-per-class.png){:class="img-responsive"}{:height="100%" width="100%"}


The table shows the model's overall performance.


|                | Confidence    |  Accuracy     |
| :------------- | :----------:  | :----------:  | 
|  Top 1         | 0.69          | 0.84          |
|  Top 5         | 0.63          | 0.97          |

Top 1 means that the predicted class is the correct class. Top 5 means that the correct class is among the 5 predicted classes with the highest score.