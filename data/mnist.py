from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class Mnist:
    def __init__(self, batch_size=64):
        # Prepare dataset
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # This transformation aims to transfer the original pil pictures into tensors
        # Elements of which are scalars in [0, 1], and the distribution is characterised by average 0.1307 and variance 0.3081
        # Use DataLoader to create batches from training dataset and the testing dataset
        train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=self.transform)
        self.train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
        # Dataset for test does not need to be trained or shuffled
        test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=self.transform)
        self.test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

    def get_train(self):
        return self.train_loader

    def get_test(self):
        return self.test_loader


# print(Mnist().get_train().dataset.__getitem__(10))
