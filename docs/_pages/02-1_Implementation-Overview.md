---

layout: page
title: Overview
permalink: /implementation/
---

The implementation is structured by the attack method. Each is implemented in a separate notebook. The first notebook `00_Helper-Functions.ipynb` contains functions that are required by multiple methods. Copies of these functions are available as modules in `modules/helper.py` and `modules/dataset.py`. The PyTorch library is used for the implementations.

The available notebooks are:
- [`00_Helper-Functions.ipynb`](https://github.com/daved01/Adversarial_Examples/blob/master/00_Helper-Functions.ipynb)
- [`01_Data_Exploration.ipynb`](https://github.com/daved01/Adversarial_Examples/blob/master/01_Data_Exploration.ipynb)
- [`02_Fast-Gradient-Sign-Method.ipynb`](https://github.com/daved01/Adversarial_Examples/blob/master/02_Fast-Gradient-Sign-Method.ipynb)
- [`03_Basic-Iterative-Method.ipynb`](https://github.com/daved01/Adversarial_Examples/blob/master/03_Basic-Iterative-Method.ipynb)
- [`04_Iterative-Least-Likely-Class-Method.ipynb`](https://github.com/daved01/Adversarial_Examples/blob/master/04_Iterative-Least-Likely-Class-Method.ipynb)
- [`05_DeepFool.ipynb`](https://github.com/daved01/Adversarial_Examples/blob/master/05_DeepFool.ipynb)

Most functions are implemented in modules which are imported into the notebooks. The modules are:

- `dataset` - Datset functions
- `helper` - Contains functions which are used by all attack methods
- `fgsm` - FGSM attack specific functions
- `bim`- BIM attack specific functions
- `illm`- ILLM attack specific functions

The functions are explained in the following sections.

To follow along with the implementations we recommend to clone the [repository](https://github.com/daved01/Adversarial_Examples) and download the data from Kaggle.


## Model
As model we use a pre-trained GoogLeNet Inception v1 model architecture [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842). It is a 22 layer (when not counting pooling) deep neural net with inception blocks. It was trained on the ImageNet dataset and can be directly imported from the PyTorch [library](https://pytorch.org/docs/stable/torchvision/models.html?highlight=googlenet#torchvision.models.googlenet). In the [ImageNet competition of 2014](http://www.image-net.org/challenges/LSVRC/2014/results) this architecture achieved the lowest classification error in the category classification and localization with provided training data.


## Data
To assess the impact of adversarial examples, a dataset with a large number of classes is preferred. The ImageNet dataset contains 1000 classes. However, instead of using the 100,000 images for testing, in this project a similar dataset is used from the *NIPS 2017: Non-targeted Adversarial Attack* challenge hosted on [Kaggle](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack). It consists of 1000 images and can be handled on a CPU. A Kaggle account is required to access it.

As required by the model, we scale and normalize the data:

{% highlight python %}
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]   

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
{% endhighlight %}

We define the dataloader object in the module `dataset`:

{% highlight python %}
data_loader = torch.utils.data.DataLoader(
    ImageNetSubset("data/ImageNet_subset/dev_dataset.csv", 
    "data/ImageNet_subset//images/", transform=preprocess))
{% endhighlight %}


## Module Helper

The module `helper` contains functions which are not attack-method specific.


{% highlight python%}
def idx_to_name(class_index):
    '''
    Converts the output class index from the googleNet to the respective name.
    
    Input:
    class_index  -- Class index as integer
    
    Returns:
    name         -- Class names corresponding to idx as string
    '''
    
    # Load dictionary from file    
    names = pd.read_csv("./data/ImageNet_subset/categories.csv")
    
    # Retrieve class name for idx
    name = names.iloc[class_index]["CategoryName"]
    
    return name
{% endhighlight %}


{% highlight python%}
def show_tensor_image(tensor):
    '''
    De-normalizes an image as a tensor and converts it back into an 8bit image object.
    
    Inputs:
    tensor -- PyTorch tensor of shape (1, 3, 224, 224)
    
    Returns:
    image  -- De-normalized image object
    '''
{% endhighlight %}


{% highlight python%}
def predict(model, image, target_label, return_grad=False):
    '''
    Predicts the class of the given image and compares the prediction with the provided label.
    
    Inputs:
    model             -- net
    image             -- Input image as tensor of shape (1, 3, 224, 224)
    target_label      -- Target label as tensor of shape (1)
    return_grad       -- Returns gradient if set True
    
    Returns:
    predicted_classes -- Numpy array of top 5 predicted class indices
    confidences       -- Numpy array of top 5 confidences in descending order
    gradient          -- None if return_grad=False. Otherwise the gradient from the prediction
                         as a tensor of shape ().
    '''      
{% endhighlight %}


{% highlight python%}
def summarize_attack(image_clean, image_adv, conf_clean, conf_adv, label_clean, label_adv, label_target, idx,
                    folder=None):
    '''
    Summarizes attack by printing info and displaying the image along with the adversary and the isolated
    perturbance. Saves image to the folder.
    
    Inputs:
    image_clean     -- Clean image as tensor of shape (1, 1, 28, 28)
    image_adv       -- Adversarial image as tensor of shape (1, 1, 28, 28)
    conf_clean      -- Confidence for the clean image
    conf_adv        -- Confidence for the adversarial image
    label_clean     -- Predicted label from the clean image
    label_adv       -- Predicted label from the adversarial image
    label_target    -- Target label as tensor of shape (1)
    idx             -- Sample index used for filename of plot export
    folder          -- If not None folder to which the image is saved.
    '''
{% endhighlight %}