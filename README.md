# Generative Adversarial Networks Compression
The final project for Columbia University COMS 6998 012 Practical Deep Learning Systems Performance

This project aims at finding an approach for Generative Adversarial Networks (GANs) compression.

We use knowledge distillation method to solve this problem by adding a pixelwise constraint on the output of the student generator and teacher generator.

![image](https://user-images.githubusercontent.com/120711627/208582366-a4816226-7f8f-479a-b670-89bab553a9e0.png)

The repository contains four parts: data, results, python file and jupyter notebooks. The results folder contains multiple trained image result and networks. We use DCGan with 512 channel in the first layer in generator as the default setting.

To use this repository, you can open the jupyter notebooks in /notebooks directly to see the result of each conditional GAN training w or w/o knowledge distillation from a larger model.

If you want to execute .py file, just run 

`train_without_distillation.py` and `train_with_distillation.py` directly, then check the result in ./py_output/cgan and ./py_output/cgan_distill

To make a comparison, we can compare the following training result:

Teacher model (channel = 512 for the first layer of G)
![16715110053331](https://user-images.githubusercontent.com/120711627/208584829-50b5b141-efe9-40a0-a104-4117a052fca3.gif)


Student model (channel = 32 for the first layer of G)
![distill_4](https://user-images.githubusercontent.com/120711627/208585488-8364ec91-b95b-45ea-9ea1-5072d04a66b5.gif)

Raw model (channel = 32 for the first layer of G)
![no_distill_4](https://user-images.githubusercontent.com/120711627/208585556-be754711-1286-4d0f-b537-d4d4e4f7f687.gif)

The raw model is trained from scratch without distillation.

After knowledge distillation, the model can preserve most of the performance of the teacher model while extremely reduce the model parameters.

                            Generator         Discriminator
                            
Teacher model (c=512)         14.3 MB              2.6 MB

Student model (c=64)           618 KB               49 KB

Student model (c=32)           271 KB               17 KB

It has 153 times compressed the discriminator and 52.75 times compressed the generator.
