# Implementations of Image to Image Translation via Generative Models using PyTorch

### 0. Introduction
This repository contains implementations of fundamental Image-to-Image Translation via Generative Models, including Pix2Pix, DiscoGAN, CycleGAN, BicycleGAN, and StarGAN. </br>
<br> Please note that I focused on implementation rather than deriving the best results. In other words, a set of hyper-parameters that I used may not produce the best results. For example, you can expect better CycleGAN results when increasing total epochs to 200.

<img src = './Introduction.PNG'>

### 1. [Pix2Pix](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/1.%20Pix2Pix)

<img src = './1. Pix2Pix/results/inference/Pix2Pix_Results_001.png'>

### 2. [DiscoGAN](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/2.%20DiscoGAN)

<img src = './2. DiscoGAN/results/inference/DiscoGAN_Edges2Shoes_Results_001.png'>

### 3. [CycleGAN](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/3.%20CycleGAN)

<img src = './3. CycleGAN/results/inference/Horse2Zebra/CycleGAN_Horse2Zebra_Results_075.png'>
<img src = './3. CycleGAN/results/inference/Zebra2Horse/CycleGAN_Zebra2Horse_Results_063.png'>

### 4. [BicycleGAN](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/4.%20BicycleGAN)

<img src = './4. BicycleGAN/results/inference/BicycleGAN_Edges2Handbags_Results_001.png'>


### 5. [StarGAN](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/5.%20StarGAN)

<img src = './5. StarGAN/results/inference/StarGAN_Aligned_CelebA_Results_0001.png'>

### 6. [Unsupervised Attention-Guided GAN](https://github.com/hee9joon/Image-to-Image-Translation/tree/master/5.%20StarGAN)

<img src = './6. Unsupervised Attention-Guided GAN/results/inference/Horse2Zebra/UAG-GAN_Horse2Zebra_Results_031.png'>
<img src = './6. Unsupervised Attention-Guided GAN/results/inference/Zebra2Horse/UAG-GAN_Zebra2Horse_Results_024.png'>

### Development Environment
- Ubuntu 18.04 LTS
- NVIDIA GFORCE GTX 1080 ti
- CUDA 10.2
- torch 1.5.1
- torchvision 0.5.0
- etc
