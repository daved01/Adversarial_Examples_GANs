---

layout: page
title: 3. Perceptibility

---


Finally, we look at perceptibility of the aforementioned attacks. In figure 18 you can see the unmodified image on the left and attacks with FGSM, BIM and ILLM. To make them comparable we attack with FGSM first and choose the highest adversarial confidence, which is around 24% here. Next, we attack with BIM and find the first adversary for which the model has a similar confidence, also 24%. It achieves this at an epsilon of 4 instead of 20 which makes this attack much less noticeable. For ILLM we cannot find an adversary with similar confidence. Instead, we choose an epsilon of 20 as for FGSM.

{% include image.html file="ILLM-comparison_all_methods.png" description="Figure 18: Attacks with FGSM (second from left), BIM (third) and ILLM (right). While epsilon for FGSM is 20, BIM achieves a similar adversarial confidence of 24% with an epsilon of just 4. Since for ILLM the adversarial confidence is consistently low, we chose epsilon of 20 for the right image."%}

Subjectively, BIM generates the least perceptible adversary. At the same time it also generates the most confident adversary with around 60% for an epsilon of 12. We have seen this for other images as well. BIM was for example able to change a correct, 90% confident detection of a fly to a 100% confident prediction of a bow (sample 894).