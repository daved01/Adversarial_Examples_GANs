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
def apply_BIM(model, mean, std, image, label, alpha, epsilon, num_iterations=10):
    '''
    Applies given number of steps of the Basic Iterative Method (BIM) attack on the input image.
    
    Inputs:
    model          -- Network under attack
    image          -- Image data as tensor of shape (1, 3, 224, 224)
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    label          -- Label from image as numpy array
    alpha          -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255.
    epsilon        -- Hyperparameter for sign method. Has to be scaled to epsilon/255.
    num_iterations -- Number of iterations to perform. Default is 10. It is recommended to use the heuristic from the
                      paper "Adversarial Examples in the Pysical World" to determine the number of iterations.
    
    Returns:
    image_adver    -- Adversarial image as tensor
    '''
{% endhighlight %}


In addition we use the following function:


{% highlight python  %}
def compute_all_bim(model, data_loader, predict, mean, std, epsilons, alpha, filename_ext):
    '''
    Computes top 1, top 5 accuracy and confidence for all samples using BIM 
    in data_loader for each epsilon.
    Does not filter false initial predictions.
    Saves the results as csv file to: ./results/BIM/BIM-all_samples.csv

    Inputs:
    model       -- Network under attack
    data_loader -- Pytorch data loader object
    predict     -- Predict function from module helper
    mean        -- Mean from data preparation
    std         -- Standard deviation from data preparation
    epsilons    -- List of epsilons for FGSM attack
    alpha       -- Hyperparameter for BIM. Must be provided as a scaled number alpha/255

    Returns:
    top1        -- Top 1 accuracy
    top5        -- Top 5 accuracy
    conf        -- Confidence
    ''' 

{% endhighlight %}


{% highlight python %}
def BIM_attack_with_selected_samples(min_confidence, max_confidence, data_loader, predict, model, mean, std, epsilons, alpha):
    '''
    Attacks the model with images from the dataset on which the model achieves clean predictions with
    confidences in the provided interval [min_confidence, max_confidence]. Only if the original
    prediction is correct an adversary is generated.
    
    Returns an average of the top1, top5 and confidence for all these samples.
    
    The number of iterations for the BIM attack is calculated by this function according to the heuristic
    from the authors of "Adversarial Examples in the Physical World".
    
    Inputs:
    min_confidence -- Minimum confidence to consider
    max_confidence -- Maximum confidence to consider
    model          -- Network under attack
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    epsilons       -- Hyperparameter 1 for attack. Provide scaled as epsilon/255 
    alpha          -- Hyperparameter 2 for attack. Provide scaled as alpha/255
    
    Returns:
    result         -- Dataframe with top1, top5 and confidence for prediction
    '''
    
{% endhighlight %}


{% highlight python %}
def compare_examples_bim(data_loader, mean, std, model, predict, summarize_attack, alpha, epsilon, idx, folder=None):
    '''
    Generates an example using BIM. Prints infos and plots clean and adversarial image side-by-side.
    
    Inputs:
    data_loader      -- Pytorch data loader object
    mean             -- Mean from data preparation
    std              -- Standard deviation from data preparation
    model            -- Network under attack
    predict          -- Predict function from module helper
    summarize_attack -- Function from module helper to describe attack
    alpha            -- Hyperparameter for BIM
    epsilon          -- Hyperparameter for BIM
    idx              -- Index of sample   
    folder           -- If given image will be saved to this folder
    '''
    
{% endhighlight %}