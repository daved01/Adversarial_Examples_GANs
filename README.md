# Adversarial Examples and Generative Adversarial Networks

The goal of this project is to explore the state-of-the-art in adversarial examples, defense mechanisms and GANs. The repository is split into two parts. The first section focuses on example generation, attacks and defense mechanisms. In the second part we deal with GANs. The findings are discussed [here](https://daved01.github.io/Adversarial_Examples_GANs/).


As dataset 1000 examples from [NIPS 2017: Adversarial Learning Development Set](https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set#categories.csv) are used. These images are similar to the ImageNet dataset. The data is not part of this repositiory and has to be downloaded seperatly. To access the data a Kaggle account is required.

----------------
## Structure

In the repository there is a notebook for each attack method. The notebook 

- `00_Helper-Functions.ipynb`

contains helper functions which are required in all other notebooks. Copies of these can be found in `modules/helper.py` and `modules/dataset.py`.

Additionally, this notebook contains a data exploration and predictions on the clean data.

The attack methods are:

- `01_Fast-Gradient-Sign-Method.ipynb`

- `02_Fast-Basic-Iterative-Method.ipynb`

- `03_Iterative-Least-Likely-Class-Method.ipynb`