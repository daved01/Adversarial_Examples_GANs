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

Repeat these steps for $$N$$ iterations to get the final adversary. In the results section we investigate how the number of iterations as well as the choices for epsilon and alpha influence the success of an attack. In the paper the authors suggest the following values for the hyperparameters:

- For alpha: $$\alpha = \frac{1}{255}$$
- Number of iterations: $$min(4+\frac{\epsilon}{\alpha}, 1.25 \cdot \frac{\epsilon}{\alpha} )$$

$$\alpha$$ is chosen to be one pixel intensity value. The number of iterations is calculated to ensure enough steps to allow for a pixel to reach the maximum adversarial perturbance, $$\epsilon$$

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

In addition we use the following function:


{% highlight python  %}
def single_attack_stats_BIM(data_loader, mean, std, model, predict, epsilon, alpha, sample, idx_to_name, num_iterations):
    '''
    Computes BIM attack and returns info about success.
    
    Inputs:
    data_loader    -- Pytorch data loader object
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    model          -- Network under attack   
    predict        -- Predict function from module helper   
    epsilon        -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    alpha          -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    sample         -- Index of sample 
    idx_to_name    -- Function to return the class name from a class index. From module helper
    num_iterations -- Number of iterations to perform the BIM with
    
    Returns:
    conf_adv       -- Confidence of adversary
    corr           -- Integer to indicate if predicted class is correct (1) or not (0)
    class_name_adv -- Label of adversarial class
    '''
{% endhighlight %}


{% highlight python %}
def visualize_attack_BIM(data_loader, mean, std, model, predict, epsilon, alpha, sample, summarize_attack, folder=None):
    '''
    Generates an adversary using BIM. Prints infos and plots clean, generated perturbance and resulting adversarial image side-by-side.
    
    Inputs:
    data_loader      -- Pytorch data loader object
    mean             -- Mean from data preparation
    std              -- Standard deviation from data preparation
    model            -- Network under attack   
    predict          -- Predict function from module helper   
    epsilon          -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    alpha            -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    sample           -- Index of sample 
    summarize_attack -- Function from module helper to describe attack
    folder           -- If given image will be saved to this folder
    '''
{% endhighlight %}


{% highlight python %}
def all_samples_attack_BIM(data_loader, mean, std, model, predict, epsilons, alpha, filename_ext, temp_filename, restart=None):
    '''
    Computes top 1, top 5 accuracy and confidence for all samples using BIM 
    in data_loader for each epsilon.
    Does not filter false initial predictions.
    Saves the results as csv file to: ./results/BIM/BIM-all_samples.csv

    Inputs:
    data_loader   -- Pytorch data loader object
    mean          -- Mean from data preparation
    std           -- Standard deviation from data preparation
    model         -- Network under attack   
    predict       -- Predict function from module helper   
    epsilons      -- List of hyperparameter for sign method. Has to be scaled to epsilon/255
    alpha         -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    filename_ext  -- Extension to file name
    temp_filename -- Name of temp file with intermediate results
    restart       -- List to use previous partial results. 
                     Format: [<filename>, <[remaining_epsilons]>]

    Returns:
    top1          -- Top 1 accuracy
    top5          -- Top 5 accuracy
    conf          -- Confidence
    ''' 
{% endhighlight %}


{% highlight python %}
def confidence_range_attack_BIM(data_loader, mean, std, model, predict, epsilons, alpha, min_confidence, max_confidence):
    '''
    Attacks the model with images from the dataset on which the model achieves clean predictions with
    confidences in the provided interval [min_confidence, max_confidence]. Only if the original
    prediction is correct an adversary is generated.
    
    Returns an average of the top1, top5 and confidence for all these samples.
    
    The number of iterations for the BIM attack is calculated by this function according to the heuristic
    from the authors of "Adversarial Examples in the Physical World".
    
    Inputs:
    data_loader    -- Pytorch data loader object
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    model          -- Network under attack   
    predict        -- Predict function from module helper   
    epsilons       -- List of hyperparameter for sign method. Has to be scaled to epsilon/255
    alpha          -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    min_confidence -- Minimum confidence to consider
    max_confidence -- Maximum confidence to consider
    
    Returns:
    result         -- Dataframe with top1, top5 and confidence for prediction
    '''
{% endhighlight %}


{% highlight python %}
def analyze_attack_BIM(data_loader, mean, std, model, predict, alpha, sample, epsilon_conf, show_tensor_image, idx_to_name, fixed_num_iter=None, save_plot=False, print_output=True):
    '''
    Generates 4 plots: Image, confidence over epsilon, top 5 confidence for clean image, top 5 confidence for adversarial image.
    The epsilons are: 0, 0.5/255, 1/255, 2/255, 4/255, 8/255, 12/255, 16/255, 20/255

    Inputs:
    data_loader       -- Pytorch data loader object
    mean              -- Mean from data preparation
    std               -- Standard deviation from data preparation
    model             -- Network under attack   
    predict           -- Predict function from module helper   
    alpha             -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    epsilon_conf      -- Epsilon for which to show the distribution in the last plot
    show_tensor_image -- Converts tensor to image. From helper module
    idx_to_name       -- Function to return the class name from a class index. From module helper
    fixed_num_iter    -- Fixed number of iterations for BIM. Calculates the recommended number for each epsilon if not given
    save_plot         -- Saves the plot to folder BIM if True
    print_output      -- Prints stats if True
    '''  
{% endhighlight %}


{% highlight python %}
def compute_hyperparameter_plot(data_loader, mean, std, model, predict, three_alphas, four_num_iter, sample, 
                                show_tensor_image, idx_to_name, save_plot=False, print_output=False):
    '''
    Generates 12 plots of confidences over epsilons for BIM attacks for provided combination of 
    three alphas and and four number_iterations.
    
    Rows:             Increasing number of iteratiosn from left to right
    Columns:          Increasing number of alpha from top to bottom.
    The epsilons are: 0, 0.5/255, 1/255, 2/255, 4/255, 8/255, 12/255, 16/255, 20/255

    Inputs:
    data_loader       -- Pytorch data loader object
    mean              -- Mean from data preparation
    std               -- Standard deviation from data preparation
    model             -- Network under attack   
    predict           -- Predict function from module helper   
    three_alpha       -- List of three alphas. Each has to be scaled to alpha/255
    four_num_iter     -- List of four number of iterations
    sample            -- Image to be used for attack
    show_tensor_image -- Converts tensor to image. From helper module
    idx_to_name       -- Function to return the class name from a class index. From module helper
    save_plot         -- Saves the plot as "BIM-Hyperparameter_variation_<sample>.png" to folder BIM if True
    print_output      -- Prints stats if True
    '''
{% endhighlight %}