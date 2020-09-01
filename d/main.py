import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

n_epochs = 887  # 887 for highest under 1000, 41 for just above 0.9
learning_rate = 0.2
momentum = 0.5

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)

x_train = mnist_train.data.reshape(-1, 28 * 28).float()
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 28 * 28).float()
y_test = torch.zeros((mnist_test.targets.shape[0], 10))
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1


class MNISTModel:
    def __init__(self):
        self.W = torch.rand((28 * 28, 10), requires_grad=True)
        self.b = torch.rand([1, 10], requires_grad=True)

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def logits(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = MNISTModel()

optimizer = torch.optim.SGD([model.W, model.b], lr=learning_rate, momentum=momentum)

print("Before training, Loss: %s, Accuracy: %s" % (model.loss(x_test, y_test).item(),
                                                   model.accuracy(x_test, y_test).item()))

epoch_accuracy = np.zeros((n_epochs, 2))

for epoch in range(n_epochs):
    model.loss(x_train,
               y_train).backward()
    optimizer.step()
    optimizer.zero_grad()
    acc = model.accuracy(x_test, y_test).item()
    epoch_accuracy[epoch] = np.array([epoch, model.accuracy(x_test, y_test).item()])
    print("Epoch: %i, Loss: %.5f, Accuracy: %.5f" % (epoch + 1,
                                                     model.loss(x_test, y_test).item(),
                                                     model.accuracy(x_test, y_test).item()))
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy [%]")
ax1.set_title("Accuracy per round through the data")
ax1.plot(epoch_accuracy[:, 0], epoch_accuracy[:, 1], '-')
ax1.plot([0, n_epochs], [0.9, 0.9], '-', c='red')

for i in range(10):
    plt.imsave("%i.png" % i, model.W[:, i].reshape(28, 28).detach(), cmap='gray')

plt.show()
