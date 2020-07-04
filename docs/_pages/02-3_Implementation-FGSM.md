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

One downside of the FGSM is that the manipulated images are often perceptible for humans for anything but the smallest changes in pixel values. This may be because this method can only modify pixel values upwards or downwards a constant value rather than a seemingly random value. Additionally, manipulations using this technique are particularly noticeable around the darker areas of an image because the relative magnitude of manipulation compared to the original image's pixel values. This can be improved by using iterative methods.

The notebook is available <a id="raw-url" href="https://raw.githubusercontent.com/daved01/Adversarial_Examples/master/01_Fast-Gradient-Sign-Method.ipynb" download>here</a>.

We first load and preprocess the data as required:

{% highlight python %}
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

{% highlight python %}
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
{% endhighlight %}

Furthermore we use the following functions:

{% highlight python %}
def visualize_attack_FGSM(data_loader, mean, std, model, predict, epsilon, sample, summarize_attack, folder=None):
    '''
    Generates an example using FGSM. Prints infos and plots clean, generated perturbance and resulting 
    adversarial image side-by-side.
    
    Inputs:
    data_loader      -- Pytorch data loader object
    mean             -- Mean from data preparation
    std              -- Standard deviation from data preparation
    model            -- Network under attack   
    predict          -- Predict function from module helper   
    epsilon          -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    sample           -- Index of sample 
    summarize_attack -- Function from module helper to describe attack
    folder           -- If given image will be saved to this folder
    '''
{% endhighlight %}


{% highlight python %}
def get_attack_series(data_loader, mean, std, model, predict, epsilons, sample, show_tensor_image, save=False):
    '''
    Generates four adversaries with the specified epsilons and displays them along with the clean image.
    
    Inputs:
    data_loader       -- Pytorch data loader object
    mean              -- Mean from data preparation
    std               -- Standard deviation from data preparation
    model             -- Network under attack   
    predict           -- Predict function from module helper   
    epsilon           -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    sample            -- Index of sample 
    show_tensor_image -- Converts tensor to image. From helper module
    save              -- Saves image series to folder FGSM is True

    Returns:
    output            -- Generated image series
    '''
{% endhighlight %}


{% highlight python %}
def confidence_range_attack_FGSM(data_loader, mean, std, model, predict, min_confidence, max_confidence):
    '''
    Attacks the model with images from the dataset on which the model achieves clean predictions with
    confidences in the provided interval [min_confidence, max_confidence]. Only if the original
    prediction is correct an adversary is generated.
    Returns an average of the top1, top5 and confidence for all these samples.
    
    Inputs:
    data_loader    -- Pytorch data loader object
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    model          -- Network under attack   
    predict        -- Predict function from module helper   
    min_confidence -- Minimum confidence to consider
    max_confidence -- Maximum confidence to consider
    
    Returns:
    result         -- Dataframe with top1, top5 and confidence for prediction
    '''
{% endhighlight %}


{% highlight python %}
def single_attack_stats_FGSM(data_loader, mean, std, model, predict, epsilon, sample, idx_to_name):
    '''
    Computes FGSM attack and returns info about success.
    
    Inputs:
    data_loader    -- Pytorch data loader object
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    model          -- Network under attack   
    predict        -- Predict function from module helper   
    epsilon        -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    sample         -- Index of sample 
    idx_to_name    -- Function to return the class name from a class index. From module helper
    
    Returns:
    conf_adv       -- Confidence of adversary
    corr           -- Integer to indicate if predicted class is correct (1) or not (0)
    class_name_adv -- Label of adversarial class
    '''
{% endhighlight %}


{% highlight python %}
def analyze_attack_FGSM(data_loader, mean, std, model, predict, sample, epsilon_conf, show_tensor_image, 
idx_to_name, save_plot=False):
    '''
    Generates 4 plots: Image, conf over epsilon, top 5 conf for clean image, top 5 conf for adversarial image.
    
    Inputs:
    data_loader       -- Pytorch data loader object
    mean              -- Mean from data preparation
    std               -- Standard deviation from data preparation
    model             -- Network under attack   
    predict           -- Predict function from module helper   
    sample            -- Index of sample
    epsilon_conf      -- Epsilon for which to show the distribution in the last plot
    show_tensor_image -- Converts tensor to image. From helper module
    idx_to_name       -- Function to return the class name from a class index. From module helper
    save_plot         -- Saves the plot to folder FGSM if True
    ''' 
{% endhighlight %}


{% highlight python %}
def all_samples_attack_FGSM(model, data_loader, predict, mean, std, epsilons):
    '''
    Computes top 1, top 5 accuracy and confidence for all samples in data_loader for each epsilon.
    Does not filter false initial predictions.
    Saves the results as csv file to: ./results/FGSM-all_samples.csv

    Inputs:
    model       -- Neural net to attack
    data_loader -- Pytorch data loader object
    predict     -- Predict function
    mean        -- Mean used in data preprocessing
    std         -- Standard deviation used in data preprocessing
    epsilons    -- List of epsilons for FGSM attack

    Returns:
    top1        -- Top 1 accuracy
    top5        -- Top 5 accuracy
    conf        -- Confidence
    '''
{% endhighlight %}


{% highlight python %}
def iterate_epsilons_FGSM(data_loader, mean, std, model, predict, sample, idx_to_name, single_attack_stats_FGSM):
    '''
    For a given sample number generates
    
    Inputs:
    data_loader              -- Pytorch data loader object
    mean                     -- Mean from data preparation
    std                      -- Standard deviation from data preparation
    model                    -- Network under attack   
    predict                  -- Predict function from module helper   
    sample                   -- Index of sample
    idx_to_name              -- Function to return the class name from a class index. From module helper
    single_attack_stats_FGSM -- Attacks model and returns stats
    '''
{% endhighlight %}


In the following section we investigate how the FGSM attack performs.