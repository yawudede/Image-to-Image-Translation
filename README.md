# Implementations of Image to Image Translation via Generative Models using PyTorch

### 0. Introduction
This repository contains implementations of fundamental Image-to-Image Translation via Generative Models, including Pix2Pix, DiscoGAN, CycleGAN, BicycleGAN, and StarGAN, Unsupervised Attention-Guided GAN, and MUNIT. </br>
<br> Please note that I focused on implementation rather than deriving the best results. In other words, a set of hyper-parameters that I used may not produce the best results. For example, you can expect better CycleGAN results when increasing total epochs to 200.

<img src = './Intro.PNG'>

### 1. [Pix2Pix : Image-to-Image Translation with Conditional Adversarial Networks](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/1.%20Pix2Pix)

<img src = './1. Pix2Pix/results/inference/Pix2Pix_Results_001.png'>

### 2. [DiscoGAN : Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/2.%20DiscoGAN)

<img src = './2. DiscoGAN/results/inference/DiscoGAN_Edges2Shoes_Results_001.png'>

### 3. [CycleGAN : Image-to-Image Translation with Conditional Adversarial Networks](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/3.%20CycleGAN)

<img src = './3. CycleGAN/results/inference/Horse2Zebra/CycleGAN_Horse2Zebra_Results_075.png'>
<img src = './3. CycleGAN/results/inference/Zebra2Horse/CycleGAN_Zebra2Horse_Results_063.png'>

### 4. [BicycleGAN : Toward Multimodal Image-to-Image Translation](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/4.%20BicycleGAN)

<img src = './4. BicycleGAN/results/inference/BicycleGAN_Edges2Handbags_Results_001.png'>


### 5. [StarGAN : Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/5.%20StarGAN)

<img src = './5. StarGAN/results/inference/StarGAN_Aligned_CelebA_Results_0001.png'>

### 6. [Unsupervised Attention-Guided GAN : Unsupervised Attention-guided Image to Image Translation](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/6.%20Unsupervised%20Attention-Guided%20GAN)

<img src = './6. Unsupervised Attention-Guided GAN/results/inference/Horse2Zebra/UAG-GAN_Horse2Zebra_Results_031.png'>
<img src = './6. Unsupervised Attention-Guided GAN/results/inference/Zebra2Horse/UAG-GAN_Zebra2Horse_Results_024.png'>

### 7. [MUNIT : Multimodal Unsupervised Image-to-Image Translation](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/7.%20MUNIT)

<img src = './7. MUNIT/results/inference/ex_guided/MUNIT_Edges2Shoes_Ex_Guided_Results_007.png'>

### 8. [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1611.07004)

<img src = './8. U-GAT-IT/results/samples/U-GAT-IT_Samples_Epoch_049.png'>

### Development Environment
```
- Ubuntu 18.04 LTS
- NVIDIA GFORCE GTX 1080 ti
- CUDA 10.2
- torch 1.5.1
- torchvision 0.5.0
- etc
```
