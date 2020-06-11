---

layout: page
title: Iterative Least Likely Method

---

### Iterative Least Likely Method (ILLM)
Both of the previous methods only try (they don’t guarantee a false classification) to change the prediction to a different class. When attacking a classifier with a lot of similar classes this can lead to uninteresting results. For example, one dog breed would be classified as another dog breed but not as a cat. The *Iterative Least Likely Class Method* (ILLM) looks at the prediction on a clean image and modifies it to output the least likely class [2].

Similar to the BIM the steps are:

Initialize with the clean image for iteration $$N=0$$

\begin{equation}
\tag{3.1}
\widetilde{X}_{0} = X 
\end{equation}

Compute fast step:

\begin{equation}
\tag{3.2}
X^{\prime}\_{1} = \widetilde{X}\_{0} - \alpha sign(\nabla\_{X} J(\widetilde{X}\_{0}, Y\_{LL}))
\end{equation}

Clip pixel values:

\begin{equation}
\tag{3.3}
\widetilde{X}\_{1} = min \( 255, X + \epsilon, max \( 0, X-\epsilon, X^{\prime}\_{1} \)\)
\end{equation}

Repeat for $$N=1$$