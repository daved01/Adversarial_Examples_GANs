---

layout: page
title: Overview
permalink: /implementation/
---

The implementation is structured by the attack method. Each is implemented in a separate notebook. The first notebook `00_Helper-Functions.ipynb` contains functions that are required by multiple methods. Copies of these functions are available as modules in `modules/helper.py` and `modules/dataset.py`. The PyTorch library is used for the implementations.

The available notebooks are:
- [`00_Helper-Functions.ipynb`](https://github.com/daved01/Adversarial_Examples/blob/master/00_Helper-Functions.ipynb)
- [`01_Fast-Gradient-Sign-Method.ipynb`](https://github.com/daved01/Adversarial_Examples/blob/master/01_Fast-Gradient-Sign-Method.ipynb)
- [`02_Fast-Basic-Iterative-Method.ipynb`](https://github.com/daved01/Adversarial_Examples/blob/master/02_Fast-Basic-Iterative-Method.ipynb)
- [`03_Iterative-Least-Likely-Class-Method.ipynb`](https://github.com/daved01/Adversarial_Examples/blob/master/03_Iterative-Least-Likely-Class-Method.ipynb)

To follow along with the implementations we recommend to clone the [repository](https://github.com/daved01/Adversarial_Examples) and download the data from Kaggle.


## Model
As model a pre-trained GoogLeNet Inception v1 model architecture is used. It is a 22 layer (when not counting pooling) deep neural net with inception blocks [???]. It can be directly imported from the PyTorch [library](https://pytorch.org/docs/stable/torchvision/models.html?highlight=googlenet#torchvision.models.googlenet). In the ImageNet competition of 2014 (ILSVRC 2014) this architecture has won the 1st price with an accuracy of $$93.3$$%.


### Data
To assess the impact of adversarial examples, a dataset with a large number of classes is preferred. The ImageNet dataset contains 1000 classes. However, instead of using the 100,000 images for testing, in this project a similar dataset is used from the *NIPS 2017: Non-targeted Adversarial Attack* challenge hosted on [Kaggle](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack). It consists of 1000 images and can be handled on a CPU. A Kaggle account is required to access it.

As required by the model, the data is preprocessed:

{% highlight python linenos %}
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]   

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

data_loader = torch.utils.data.DataLoader(
    ImageNetSubset("data/ImageNet_subset/dev_dataset.csv", 
    "data/ImageNet_subset//images/", transform=preprocess))
{% endhighlight %}