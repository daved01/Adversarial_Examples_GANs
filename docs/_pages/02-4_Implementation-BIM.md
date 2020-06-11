---

layout: page
title: Basic Iterative Method

---

### Basic Iterative Method (BIM)
An extension of the FGSM is BIM. It applies the FGSM multiple times to an image with step size $$\alpha$$ and clips the resulting pixel values to ensure that they stay similar to the original ones [2].

Iterative methods like the BIM are slower but generally produce more subtle perturbation to images.

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

For the hyperparameter [2] suggest:

- $$\alpha$$ = 1
- Number of iterations: $$min(4+\epsilon, 1.25 \cdot \epsilon)$$