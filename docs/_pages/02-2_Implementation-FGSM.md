---

layout: page
title: Fast Gradient Sign Method

---

The earliest and simplest method to generate adversarial examples is the Fast Gradient Sign Method (FGSM) as introduced in [Explaining and Harnessing Adversarial Examples](http://arxiv.org/abs/1607.02533) by Goodfellow, I. et al. This non-iterative method generates examples in one step and leads to robust adversaries. It computes a step of gradient descent and moves one step of magnitude $$\epsilon$$ into the direction of this gradient:

\begin{equation}
\tag{1.1}
\widetilde{X} = X + \eta
\end{equation}

\begin{equation}
\tag{1.2}
\eta = \epsilon sign(\nabla_{x} J(\Theta, x, y))
\end{equation}

Essentially, FGSM takes one step to increase the cost function with the correct label, hoping that this will be enough to change the top prediction. The main benefit of this technique is its speed

One downside of the FGSM is that the manipulated images are often perceptible for humans for anything but the smallest changes in pixel values. This may be because this method can only modify pixel values upwards or downwards a constant value rather than a seemingly random value. Additionally, manipulations using this technique are particularly noticeable around the darker areas of an image because the relative magnitude of manipulation compared to the original image's pixel values. This can be improved by using iterative methods.

The notebook is available <a id="raw-url" href="https://raw.githubusercontent.com/daved01/Adversarial_Examples/master/01_Fast-Gradient-Sign-Method.ipynb" download>here</a>.

We first load and preprocess the data as previously explained. The attack is implemented as:

{% highlight python linenos %}
def attack_FGSM(mean, std, image, epsilon, grad_x):
    '''
    Applies Fast Gradient Sign Method (FGSM) attack on the input image.
    
    Inputs:
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    image          -- Image data as tensor of shape (1, 3, 224, 224)  
    epsilon        -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    grad_x         -- Gradient obtained from prediction with image on model
    
    Returns:
    image_tilde    -- Adversarial image as tensor
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
    
    return image_tilde
{% endhighlight %}

We have to provide $$\epsilon$$ normalized with 255. Since we standardize and scale the data in the data preparation we have to divide epsilons channel-wise by the standard deviation (line 17). We clip the values to the range of the original image in line 27, which is equivalent to keeping the values in the range of [0, 1] for an image which has not been standardized and scaled.

We investigate how the FGSM attack performs in the Results section.