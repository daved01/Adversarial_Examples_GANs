---

layout: page
title: Basic Iterative Method

---

The Basic Iterative Method (BIM) is a simple extension of the Fast Gradient Sign Method, where instead of have one large step, it applies the FGSM multiple times to an image with step size $$\alpha$$. The resulting adversarial example can then be clipped to limit the maximum perturbance for each pixel [Adversarial Examples in the Physical World]((http://arxiv.org/abs/1607.02533)).

Iterative methods like the BIM are slower, but generally produce more subtle perturbation to images.

The steps are:

Initialize with the clean image $$X$$ for iteration $$N=0$$

\begin{equation}
\tag{2.1}
\widetilde{X}_{0} = X 
\end{equation}

Compute fast step:

\begin{equation}
\tag{2.2}
X^{\prime}\_{1} = \widetilde{X}\_{0} + \alpha sign(\nabla\_{X} J(\widetilde{X}\_{0}, Y\_{true}))
\end{equation}

Clip pixel values:

\begin{equation}
\tag{2.3}
\widetilde{X}\_{1} = min \( 255, X + \epsilon, max \( 0, X-\epsilon, X^{\prime}\_{1} \)\)
\end{equation}

Repeat for $$N=1$$

For the hyperparameter the authors suggest:

- For alpha: $$\alpha = \frac{1}{255}$$
- Number of iterations: $$min(4+\frac{\epsilon}{\alpha}, 1.25 \cdot \frac{\epsilon}{\alpha} )$$


## Functions

We implement BIM as follows:

{% highlight python %}
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
{% endhighlight %}


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
def visualize_attack_BIM(data_loader, mean, std, model, predict, epsilon, alpha, sample, summarize_attack,folder=None):
    '''
    Generates an example using BIM. Prints infos and plots clean, generated perturbance and resulting adversarial image side-by-side.
    
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
def analyze_attack_BIM(data_loader, mean, std, model, predict, alpha, sample, epsilon_conf, show_tensor_image, idx_to_name, 
num_iterations=None, save_plot=False, print_output=True):
    '''
    Generates 4 plots: Image, conf over epsilon, top 5 conf for clean image, top 5 conf for adversarial image.
    
    Inputs:
    data_loader       -- Pytorch data loader object
    mean              -- Mean from data preparation
    std               -- Standard deviation from data preparation
    model             -- Network under attack   
    predict           -- Predict function from module helper   
    alpha             -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    epsilon_conf      --
    show_tensor_image -- 
    num_iterations    --
    save_plot         --
    print_output      -- 
    '''  
{% endhighlight %}