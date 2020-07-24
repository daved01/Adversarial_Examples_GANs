---

layout: page
title: 4. Conclusions

---



Starting with the simplest adversarial attack, the Fast Gradient Sign Method (FGSM). It is fairly computationally inexpensive attack that worked against most of the sample images. FGSM, however, has difficulties against some of the more difficult images with higher initial confidences. This attack may have its place in quickly generating large quantities of adversarial examples.

FGSM is fast, but not always effective. Works most of the time.

BIM is a more effective version of FGSM by completing the calculations iteratively. It was able to succeed in 100% of the samples with very little perturbance.

DeepFool is even more effective in creating adversarial examples with even less perturbance than BIM. 

# Further Reading and Other Work
