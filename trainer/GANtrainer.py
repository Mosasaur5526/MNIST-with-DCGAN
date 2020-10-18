import torch
import torch.nn as nn
import torch.optim as optim
import utils.utils as utils
from model.generator import Generator
from model.discriminator import Discriminator
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


class Trainer:
    def __init__(self, nc=1, nz=100, ngf=64, ndf=64, lr=0.0002, beta1=0.5, ngpu=1):
        self.nz = nz
        self.dataloader = None
        self.img_list = None
        self.G_losses = None
        self.D_losses = None
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        self.netG = Generator(nz, ngf, nc).to(self.device)
        self.netG.apply(utils.weights_init)
        self.netD = Discriminator(nc, ndf).to(self.device)
        self.netD.apply(utils.weights_init)
        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)
        self.real_label = 1.
        self.fake_label = 0.
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Set the dataloader
    def load_data(self, dataloader):
        self.dataloader = dataloader

    # Train the networks
    def train(self, num_epochs, render=True):

        # Check whether the dataloader is None, if so quit the training
        if self.dataloader is None:
            print("Data has not been loaded yet!")
            return

        # Else start the training
        print("Start training loop...")
        self.netG.apply(utils.weights_init)
        self.netD.apply(utils.weights_init)
        self.img_list = []
        self.G_losses = []
        self.D_losses = []

        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                # Train the discriminator with real images
                self.netD.zero_grad()
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                output = self.netD(real_cpu).view(-1)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # Train the discriminator with fake images
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                output = self.netD(fake.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                # Update the parameters
                errD = errD_real + errD_fake
                self.optimizerD.step()

                # Train the generator
                self.netG.zero_grad()
                label.fill_(self.real_label)
                output = self.netD(fake).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                if i % 100 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(self.dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

            with torch.no_grad():
                # Fixed noise is employed here, because I want to generate images of same digits for more conspicuous comparisons
                fake = self.netG(self.fixed_noise).detach().cpu()
            self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            # Plot the training result of this epoch
            if render:
                self.draw_current_image()

    # Draw the original image
    def draw_original_image(self):
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()

    # Plot the loss curves of both the generator and the discriminator
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    # Draw the last 64 figures
    def draw_current_image(self):
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.img_list[-1], (1, 2, 0)))
        plt.show()
