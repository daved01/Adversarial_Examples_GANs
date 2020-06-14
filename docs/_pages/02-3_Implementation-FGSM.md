---

layout: page
title: Fast Gradient Sign Method

---

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

The notebook is available <a id="raw-url" href="https://raw.githubusercontent.com/daved01/Adversarial_Examples/master/01_Fast-Gradient-Sign-Method.ipynb" download>here</a>.

We first load and preprocess the data as required:

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
    ImageNetSubset("data/ImageNet_subset//dev_dataset.csv", "data/ImageNet_subset//images/", transform=preprocess))
{% endhighlight %}


The attack is implemented as:

{% highlight python linenos %}
'''
    Generates adversarial image from the input image using the Fast Gradient Sign Method (FGSM).
    
    Inputs:
    image       -- Image data as tensor
    epsilon     -- Hyperparameter
    grad_x      -- Gradient of the cost with respect to x
    
    Returns:
    image_tilde -- Adversarial image as tensor
    '''
    
    ## Calculated normalized epsilon and convert it to a tensor
    eps_normed = [epsilon/s for s in std]
    eps_normed = torch.tensor(eps_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    ## Compute eta part
    eta = eps_normed * grad_x.sign()
    
    ## Apply perturbation
    image_tilde = image + eta    
    
    ## Clip image to maintain the range [min, max]
    image_tilde = torch.clamp(image_tilde, image.detach().min(), image.detach().max())
    
    ## Calculate normalized range [0, 1] and convert them to tensors
    zero_normed = [-m/s for m,s in zip(mean, std)]
    zero_normed = torch.tensor(zero_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    max_normed = [(1-m)/s for m,s in zip(mean,std)]
    max_normed = torch.tensor(max_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    ## Clip image so after denormalization and destandardization, the range is [0, 255]
    image_tilde = torch.max(image_tilde, zero_normed)
    image_tilde = torch.min(image_tilde, max_normed)
{% endhighlight %}

In the following section we investigate how the FGSM attack performs.

## Results

The following is an example of the original image, the generated perturbance and the resulting adversarial image at an epsilon of $$\frac{4}{255}$$. Subjectively, at that epsilon the attack remains only slightly perceptible. For a greater epsilon the attack become much more noticeable. The image appears increasingly noisy.

![FGSM Attack](/Adversarial_Examples_GANs/assets/Adversarial_Examples-FGSM-Attack.png){:class="img-responsive"}{:height="100%" width="100%"}

In the following we are going to analyze the effect of different attack strengths on different images. We evaluate the following hypothesis:

**Hypothesis 1:** Images with a high initial confidence are harder to manipulate.

### All Images

We consider a plot of accuracy and confidence over the attack strength epsilon. For this hypothesis to be true we would observe a sharp drop in accuracy with increasing attack strength. The higher the initial confidence is, the smaller should the slope of the accuracy be. That is it is harder to attack the network at the same epsilon. At the same time the confidence should drop slightly since more and more robust features are altered.

Recall from the section [Data Exploration]({{ site.baseurl | prepend: site.url | relative_url }}{% link _pages/02-2_Implementation-DataModel.md %}) how the confidence over all data is distributed. We consider correct initial classifications only and split the data by confidences in the interval of 5% points.

![FGSM Attack](/Adversarial_Examples_GANs/assets/Adversarial_Examples-FGSM-Confidence_Levels-99.png){:class="img-responsive"}{:height="100%" width="100%"}

In the figure above you can see how the confidence initially decreases and then slightly increases again. This L-shaped confidence distribution can be found in some individual images as well (see below). The fact that the average follows a similar pattern shows that this could be the case in the majority of images. Note that only a small fraction, up to an epsilon of around 4, the attack is only slightly perceptible. For greater epsilon the attack becomes very obvious when looking at the image.

Also interesting is the accuracy. First, it decreases strongly. At an epsilon of around 4 however, it increases again until it reaches an epsilon for around 10. This is roughly the same point where the confidence hits its minimum. After this increase it decreases monotonously for the rest of the range of the attack.

It steadily decreases. The top 5 accuracy follows the same trend with an offset, as expected. ——> How are the classes distributed? Neighbouring classes etc

![FGSM Attack](/Adversarial_Examples_GANs/assets/Adversarial-Examples-FGSM-Confidence_Levels-Accuracies.png){:class="img-responsive"}{:height="100%" width="100%"}

The figure above shows the accuracy for different initial confidence levels. It clearly shows that for lower initial confidence the drop in accuracy is stronger and levels out at lower values.

A look at the confidences shows that their trend is similar for all initial confidences. The initial decrease is stronger for lower initial values.

![FGSM Attack](/Adversarial_Examples_GANs/assets/Adversarial-Examples-FGSM-Confidence_Levels-Confidences.png){:class="img-responsive"}{:height="100%" width="100%"}

All that means that the higher the initial confidence is, the harder it is to attack the model with these samples.

### Individual Images

From the images with the highest initial confidence of over 99% we can see that it is hard to craft adversarial examples from them.
The following plot shows how for only one epsilon a false class is predicted at low confidence. Further increasing the attack interestingly outputs the correct class again, at lower confidence.

![FGSM Attack](/Adversarial_Examples_GANs/assets/Confidence_Levels-Sample766.png){:class="img-responsive"}{:height="100%" width="100%"}

Another interesting case is illustrated here.

![FGSM Attack](/Adversarial_Examples_GANs/assets/Confidence_Levels-Sample132.png){:class="img-responsive"}{:height="100%" width="100%"}

It is nor possible craft an adversarial example from this image with the FGSM. Interestingly, the confidence follows a U shape and reaches almost the initial value again.

An example for lower initial confidence of 80% is shown below.

![FGSM Attack](/Adversarial_Examples_GANs/assets/Confidence_Levels-Sample528.png){:class="img-responsive"}{:height="100%" width="100%"}

As can be seen the attack successfully causes a false prediction early on. The falsely predicted class changes while increasing epsilon. The confidence seems to be levelled out at around 15%.