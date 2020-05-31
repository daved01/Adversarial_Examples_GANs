---

layout: page
title: Implementation
permalink: /implementation/
---

The project is implemented in different notebooks. The first notebook `00_Helper-Functions.ipynb` contains functions that are required by multiple methods. Copies of these functions are available as modules in `modules/helper.py` and `modules/dataset.py`. The PyTorch library is used for the implementations.


### Model
As model a pre-trained GoogLeNet Inception v1 model architecture is used. It is a 22 layer (when not counting pooling) deep neural net with inception blocks [???]. It can be directly imported from the PyTorch [library](https://pytorch.org/docs/stable/torchvision/models.html?highlight=googlenet#torchvision.models.googlenet). In the ImageNet competition of 2014 (ILSVRC 2014) this architecture has won the 1st price with an accuracy of $$93.3$$%.


### Data
To assess the impact of adversarial examples, a dataset with a large number of classes is preferred. The ImageNet dataset contains 1000 classes. However, instead of using the 100,000 images for testing, in this project a similar dataset is used from the *NIPS 2017: Non-targeted Adversarial Attack* challenge hosted on [Kaggle](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack). It consists of 1000 images and can be handled on a CPU. A Kaggle account is required to access it.


### Predictions on the clean dataset

The goal of adversarial examples is to fool the network into prediction a wrong class with high confidence. That's why we inspect the confidence of the network on the clean data first.

In the data there are 452 out of 1000 distinct classes represented. The most frequent class is `ballplayer, baseball player` (class index 981) with 8 occurrences, followed by `racer, race car, racing car`, `stone wall` and `worm fence, snake fence, snake-rail fence, ...` with 7 each. Within these frequent classes the model's confidence is around $$58$$% with a standard deviation of around $$27$$. This is probably due to false predictions.

The average confidence for a class is between $$99.99$$% (`barrel, cask`) and $$41.64$$% (`sandal`). In the plot the distribution of the average confidence can be seen. For over half of the examples the model has an average confidence of over $$60$$% and over $$\frac{2}{3}$$ of the examples have a confidence of over $$50$$%.


![name of the image]({{site.baseurl}}/assets/Adversarial-Examples_Average-confidence-per-class.png){:class="img-responsive"}{:height="100%" width="100%"}


The table shows the model's overall performance.


|                | Confidence    |  Accuracy     |
| :------------- | :----------:  | :----------:  | 
|  Top 1         | 0.69          | 0.84          |
|  Top 5         | 0.63          | 0.97          |

Top 1 means that the predicted class is the correct class. Top 5 means that the correct class is among the 5 predicted classes with the highest score.