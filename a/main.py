import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.rcParams.update({'font.size': 11})


class SigmoidModel:
    def __init__(self):
        self.W = torch.rand((1, 1), requires_grad=True, dtype=torch.float)
        self.b = torch.rand((1, 1), requires_grad=True, dtype=torch.float)

    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    # Uses Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = SigmoidModel()

# Observed/training input and output
x_train = torch.tensor([[0.], [1.]], requires_grad=True, dtype=torch.float)
y_train = torch.tensor([[1.], [0.]], dtype=torch.float)

optimizer = torch.optim.SGD([model.W, model.b], 0.1)

for epoch in range(5000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

fig = plt.figure("Logistic regression: the logical NOT operator")

plot1 = fig.add_subplot()

plot1.plot(x_train.detach(),
           y_train.detach(),
           'o',
           label="$(\\hat x_1^{(i)},\\hat y^{(i)})$",
           color="blue")

plot1_info = fig.text(0.01, 0.02, "W = %s, b = %s, loss = %s" % (model.W[0, 0].item(), model.b[0, 0].item(), model.loss(x_train, y_train).item()))

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$y$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)

table = plt.table(cellText=[[0, 1],
                            ["${%.1f}$" % model.f(torch.tensor([[0.]]).detach()),
                             "${%.1f}$" % model.f(torch.tensor([[1.]])).detach()]],
                  colWidths=[0.1] * 3,
                  colLabels=["$x_1$", "$f(x)$"],
                  cellLoc="center",
                  loc="lower right")

values = np.arange(0., 1.1, 0.1).reshape(11, -1)
f = np.arange(0., 1.1, 0.1).reshape(11, -1)
for i in range(len(values)):
    f[i] = model.f(torch.tensor(values[i], dtype=torch.float)).detach()

plot1.plot(values, f)

fig.canvas.draw()

plt.savefig('a')
plt.show()
