# Image-to-Image-Translation
Image to Image Translation (GAN) using PyTorch

### 2. [DiscoGAN](https://arxiv.org/pdf/1703.05192.pdf)

#### 1. Networks
<img src = ./Results/DiscoGAN_Diagram.png>

#### 2. Edges2Shoes Dataset [Download](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz)

#### 3. Training Process
In the order of...
- real_A
- fake_A(Generated from Real B)
- fake_BAB(Reconstructed from fake_A) 
- real_B 
- fake_B(Generated from Real A)
- fake_ABA(Reconstructed from fake_B)
<img src = ./Results/DiscoGAN_Results_Sample.gif>

#### 4. Results (Inference After 40 Epochs)
<img src = ./Results/DiscoGAN_Results_Test.gif>

#### 5. DiscoGAN Loss over Epoch 40
<img src = ./Results/DiscoGAN_Losses_Epoch_40.png>
