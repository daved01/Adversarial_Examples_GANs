---

layout: page
title: Basic Iterative Method

---

The Basic Iterative Method (BIM) from [Adversarial Examples in the Physical World](http://arxiv.org/abs/1607.02533) is a simple extension of the Fast Gradient Sign Method, where instead of taking one large step, it applies the FGSM multiple times to an image with step size $$\alpha$$, the change in pixel value per iteration. The resulting adversary can then be clipped to limit the maximum perturbance for each pixel.

Iterative methods like the BIM are slower, but generally produce more successful and subtle perturbation to images.

First, a clean image $$X$$ is used for initialization in iteration $$N=0$$

\begin{equation}
\tag{2.1}
\widetilde{X}_{0} = X 
\end{equation}

Using this image a step similar to equation (1.2) from the FGSM is performed:

\begin{equation}
\tag{2.2}
X^{\prime}\_{1} = \widetilde{X}\_{0} + \alpha sign(\nabla\_{X} J(\widetilde{X}\_{0}, Y\_{true}))
\end{equation}

The adversarial example is then clipped to ensure that all pixel values are within the bounds of epsilon and the maximum and minimum pixel intensities:

\begin{equation}
\tag{2.3}
\widetilde{X}\_{1} = min \( 255, X + \epsilon, max \( 0, X-\epsilon, X^{\prime}\_{1} \)\)
\end{equation}

Repeat these steps for $$N$$ iterations to get the final adversary. In the results section 2.2 we investigate how the number of iterations as well as the choices for epsilon and alpha influence the success of an attack. In the paper the authors suggest the following values for the hyperparameters:

- For alpha: $$\alpha = \frac{1}{255}$$
- Number of iterations: $$min(4+\frac{\epsilon}{\alpha}, 1.25 \cdot \frac{\epsilon}{\alpha} )$$

$$\alpha$$ is chosen to be one pixel intensity value and the number of iterations is calculated to ensure enough steps to allow for a pixel to reach the maximum adversarial perturbance, $$\epsilon$$.

## Functions

We implement BIM as follows:

{% highlight python linenos %}
def attack_BIM(mean, std, model, image, class_index, epsilon, alpha, num_iterations=10):
    '''
    Applies given number of steps of the Basic Iterative Method (BIM) attack on the input image.
    
    Inputs:
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    model          -- Network under attack
    image          -- Image data as tensor of shape (1, 3, 224, 224)
    class_index    -- Label from image as numpy array   
    epsilon        -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    alpha          -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    num_iterations -- Number of iterations to perform. Default is 10. It is recommended to use the heuristic from the
                      paper "Adversarial Examples in the Pysical World" to determine the number of iterations
    
    Returns:
    image_adver    -- Adversarial image as tensor
    '''

    # Convert label to torch tensor of shape (1)
    class_index = torch.tensor([class_index])

    # Check input image and label shapes
    assert(image.shape == torch.Size([1, 3, 224, 224]))
    assert(class_index.shape == torch.Size([1]))
    
    # Initialize adversarial image as image according to equation 2.1
    image_adver = image.clone()    
    
    # Calculate normalized range [0, 1] and convert them to tensors
    zero_normed = [-m/s for m,s in zip(mean, std)]
    zero_normed = torch.tensor(zero_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    max_normed = [(1-m)/s for m,s in zip(mean,std)]
    max_normed = torch.tensor(max_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    # Calculate normalized alpha
    alpha_normed = [alpha/s for s in std]
    alpha_normed = torch.tensor(alpha_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)

    # Calculated normalized epsilon and convert it to a tensor
    eps_normed = [epsilon/s for s in std]
    eps_normed = torch.tensor(eps_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    # Calculate the maximum change in pixel value using epsilon to be later used in clip function
    image_plus = image + eps_normed
    image_minus = image - eps_normed
    
    for i in range(num_iterations):
        
        # Make a copy and detach so the computation graph can be constructed
        image_adver = image_adver.clone().detach()
        image_adver.requires_grad=True
        
        # Compute cost with example image_adversarial        
        pred = model(image_adver)        
        loss = F.nll_loss(pred, class_index)        
        model.zero_grad()        
        loss.backward()        
        grad_x = image_adver.grad.data       
        
        # Check if gradient exists
        assert(image_adver.grad is not None)
               
        # Compute X_prime according to equation 2.2
        image_prime = image_adver + alpha_normed * grad_x.detach().sign()
        assert(torch.equal(image_prime, image_adver) == False)
      
        # Equation 2.3 part 1
        third_part_1 = torch.max(image_minus, image_prime)
        third_part = torch.max(zero_normed, third_part_1)
              
        # Equation 2.3 part 2
        image_adver = torch.min(image_plus, third_part)                 
        image_adver = torch.min(max_normed, image_adver)                        

    return image_adver
{% endhighlight %}

Note that we normalize $$\alpha$$ channel-wise in line 38 since we are working with standardized and scaled images as inputs.