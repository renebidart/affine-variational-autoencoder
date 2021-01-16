# Affine Variational Autoencoder
![alt text](imgs/affine_vae_1d.png)

---
**[3D Version of this](https://github.com/renebidart/disentangled-3d)**
---

## Smaller models with better generalization:
1. Add affine layers before and after the VAE to make it generalize to affine transformed images
2. Optimize affine transforms during training so the model will learn to encode only a subset of the full distribution.

The AVAE is a variant of the VAE that can handle generalizing to affine transformed images by adding affine transform layers before and after the model, and optimizing the affine transform to reduce the VAE's loss. Because the VAE's loss is a lower bound of the probablilty of the sample, by optimizing this we push the input to be likely under the training distribution. For both rotations and general affine transforms on MNIST the AVAE significantly improves performance compared to VAE.

Also, the affine transform can be optimized during training to force the model to only learn a subset of the input distribution, enabling the model to generalize to affine transforms while only encoding a small subset of all possible ones. This introduces a trade-off between model size and compute through the optimization. 

---
### Details:
More details are given in this [short paper ](avae_icml_udl_2019.pdf) and [this blog](http://www.renebidart.com/Affine-Variational-Autoencoders-for-Efficient-Generalization/).
* The AVAE and VAE architectures are both located in [models.py](model/models.py)
* Evaluation of rotation opimization during training is in [notebooks/optimizing_rotations.ipynb](notebooks/optimizing_rotations.ipynb)
* Evaluation of rotation opimization during training is in [optimizing_batch_rotations.ipynb](notebooks/optimizing_batch_rotations.ipynb)
* An investigation into the rotation optimization is in [check_rotation.ipynb](notebooks/check_rotation.ipynb)
