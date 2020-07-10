---

layout: page
title: Deep Fool

---

The [Deep Fool](https://doi.org/10.1109/cvpr.2016.282) algorithm searches for an adversary with the smallest possible perturbation by orthogonally projecting from the sample onto the closest decision boundary and iteratively moving past it. According to the authors this method generates very subtle perturbations in contrast to the coarse approximations of the optimal perturbation vectors generates by FGSM.

Figure 1 shows the concept behind DeepFool for a linear, binary classifier.

{% include image_small.html file="DeepFool.png" description="Figure 1: DeepFool for a linear, binary classifier. From [DeepFool: a simple and accurate method to fool deep neural networks](https://doi.org/10.1109/cvpr.2016.282)"%}

For an image $$X$$ we get the minimum distance $$r_{min}$$ to the decision boundary $$F$$ by:

\begin{equation}
\tag{4.1}
r_{i}(X_{i}) = - \frac{F(X_{i})}{|| w ||_{2} } w
\end{equation}


The algorithms uses equation 4.1 to generate an optimal perturbation $$r_{opt}$$ in the following steps:

- Initialize the sample $$X_{0}$$ with the clean image $$X$$.

While $$sign( f(X_{i}) ) = sign( f(X_{0}) )$$:

- Compute $$r_{i}$$ with equation (4.1) and update  

\begin{equation}
\tag{4.2}
X_{i+1} = X_{i} + r_{i}
\end{equation}

Result: $$r_{opt}$$

The decision boundaries are linearized locally if they are non-linear. For multi-class classifier the steps above can be generalized. For details see the original paper.

We use the Python implementation which is [available](https://github.com/lts4/deepfool) from the authors and copy it to the `modules` folder. To use it we have to remove the softmax layer from the googleNet which is shown in the [Jupiter Notebook](https://github.com/daved01/Adversarial_Examples/blob/master/05_Deep_Fool.ipynb).