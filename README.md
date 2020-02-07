# Image-to-Image-Translation
Image to Image Translation (GAN) using PyTorch

### 1. Pix2Pix: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)

### 2. DiscoGAN: [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/pdf/1703.05192.pdf)

#### 1. Networks
<img src = ./Results/DiscoGAN_Diagram.PNG>

#### 2. Edges2Shoes Dataset [Download](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz)

#### 3. Sample Images during Training
In the order of,
- real_A
- fake_A   (Generated from Real B)
- fake_BAB (Reconstructed from fake_A) 
- real_B 
- fake_B   (Generated from Real A)
- fake_ABA (Reconstructed from fake_B)
<img src = ./Results/DiscoGAN_Results_Sample.gif>

#### 4. Results (Inference After 40 Epochs)
<img src = ./Results/DiscoGAN_Results_Test.gif>

#### 5. Training Loss over Epoch 40
<img src = ./Results/DiscoGAN_Losses_Epoch_40.png>

### 3. CycleGAN: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

#### 1. Networks
[Diagram Source](https://modelzoo.co/model/mnist-svhn-transfer)
<img src = ./Results/CycleGAN_Networks.png>

#### 2. Horse2Zebra Dataset [Download](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip)

#### 3. Best Sample Image during Training
In the order of,
- real A   (Horse)
- fake B   (Generated Zebra)
- fake ABA (Reconstructed Horse)
- real B   (Zebra)
- fake A   (Generated Horse)
- fake BAB (Reconstructed Zebra)
<img src = ./Results/CycleGAN_Horse2Zebra_Best_1.png>

#### 4. Best Result (Inference After 100 Epochs)
<img src = ./Results/CycleGAN_Horse2Zebra_Best_2.png>

#### 5. Training Loss over Epoch 100
<img src = ./Results/CycleGAN_Losses_Epoch_100.png>
