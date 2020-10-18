from data.mnist import Mnist
from trainer.GANtrainer import Trainer


data = Mnist(batch_size=64).get_train()
trainer = Trainer(nz=100, ngf=64, ndf=64, lr=0.0002, beta1=0.5, ngpu=1)
trainer.load_data(data)
trainer.draw_original_image()
trainer.train(num_epochs=5)
trainer.plot_loss()
