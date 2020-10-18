from data.mnist import Mnist
from trainer.GANtrainer import Trainer
import utils.gif_creator as gif


# Choose an existing path to save all the results
path = "D:"

# Set the number of training epochs
num_epochs = 10

# Acquire the data (loader)
data = Mnist(batch_size=64).get_train()

# Create a trainer
# nz: dimension of noise vector as input
# ngf: dimension of feature maps of the generator
# ndf: dimension of feature maps of the discriminator
# lr: learning rate
# beta1: the parameter for the Adam optimiser
# autosave: the directory to save all the training results
trainer = Trainer(nz=100, ngf=64, ndf=64, lr=0.0002, beta1=0.5, autosave=path)

# Get the data (loader) prepared
trainer.load_data(data)

# Draw the original image of the training dataset
trainer.draw_original_image()

# Do the training!
trainer.train(num_epochs=num_epochs, render=False)

# Plot the loss curves of both the generator and the discriminator
trainer.plot_loss()

# Create a gif which reflects the training process (may cause warnings but matters nothing)
img_list = gif.get_image_list(path, num_epochs)
gif.create_gif(img_list, path, duration=0.8)
