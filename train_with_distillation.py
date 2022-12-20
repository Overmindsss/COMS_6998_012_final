from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


class Generator(nn.Module):
    def __init__(self, nc=10, nz=100, n_out=1, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z+C, going into a convolution
            nn.ConvTranspose2d(nz+nc, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, n_out, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc+1) x 28 x 28
            nn.Conv2d(nc+1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


seed_it(1314)

dataset = dset.MNIST(root='./data', download=True,
                     transform=transforms.Compose([
                         #transforms.Resize(imageSize),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,)),
                     ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nc = 1
nz = 100
ngf = 4
ndf = 4

netG = Generator(ngf=ngf).to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator(ndf=ndf).to(device)
netD.apply(weights_init)
print(netD)

teacher_G = Generator(ngf=64).to(device)
teacher_G.load_state_dict(torch.load('./teacher/netG_epoch_24.pth'))

fixed_input = torch.load('./fixed_input.pth')

outf = './cgan_distill'
os.mkdir(outf)

GAN_criterion = nn.BCELoss()
L1_criterion = nn.L1Loss()

real_label = 1.
fake_label = 0.

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

img_list = []
G_losses = []
D_losses = []
G_gan_losses = []

for epoch in range(25):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()

        images, real_conditions = data

        real_cpu = images.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

        real_d_conditions = torch.unsqueeze(real_conditions, 1)
        real_d_conditions = torch.unsqueeze(real_d_conditions, 2)
        real_d_conditions = torch.unsqueeze(real_d_conditions, 3)
        real_d_conditions = real_d_conditions.expand(-1, -1, 28, 28)
        real_d_conditions = real_d_conditions.to(device)

        real_d_input = torch.cat([real_cpu, real_d_conditions], 1)

        output = netD(real_d_input).view(-1)
        errD_real = GAN_criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)

        fake_conditions = torch.randint(0, 10, (batch_size,)).to(device)

        fake_g_conditions = torch.unsqueeze(fake_conditions, 1).expand(batch_size, 10)
        fake_g_conditions = torch.unsqueeze(fake_g_conditions, 2)
        fake_g_conditions = torch.unsqueeze(fake_g_conditions, 3)
        fake_g_conditions = fake_g_conditions.to(device)
        fake_g_input = torch.cat([noise, fake_g_conditions], 1)

        fake = netG(fake_g_input)
        label.fill_(fake_label)

        fake_d_conditions = torch.unsqueeze(fake_conditions, 1)
        fake_d_conditions = torch.unsqueeze(fake_d_conditions, 2)
        fake_d_conditions = torch.unsqueeze(fake_d_conditions, 3)
        fake_d_conditions = fake_d_conditions.expand(-1, -1, 28, 28)
        fake_d_conditions = fake_d_conditions.to(device)

        fake_d_input = torch.cat([fake, fake_d_conditions], 1)

        output = netD(fake_d_input.detach()).view(-1)
        errD_fake = GAN_criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        output = netD(fake_d_input).view(-1)

        fake_teacher = teacher_G(fake_g_input)

        errG_gan = GAN_criterion(output, label)
        errG_reconst = L1_criterion(fake_teacher, fake)
        errG = errG_gan + errG_reconst

        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 100 == 0:
            print(
                '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\nLoss_G_gan: %.4f\tLoss_G_reconst: %.4f\n D(x): %.4f\tD(G(z)): %.4f / %.4f\n'
                % (epoch, 25, i, len(dataloader),
                   errD.item(), errG.item(),
                   errG_gan.item(), errG_reconst.item(),
                   D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        G_gan_losses.append(errG_gan.item())

    with torch.no_grad():
        fake = netG(fixed_input).detach().cpu()
    vutils.save_image(fake.detach(),
                      '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                      normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))