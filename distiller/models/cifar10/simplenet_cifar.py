@@ -21,30 +21,35 @@ __all__ = ['simplenet_cifar']


class Simplenet(nn.Module):
    """
    This is Simplenet but with only one small Linear layer, instead of two Linear layers,
    one of which is large.
    26K parameters.
    python compress_classifier.py ${MNIST_PATH} --arch=simplenet_mnist --vs=0 --lr=0.01

    ==> Best [Top1: 98.970   Top5: 99.970   Sparsity:0.00   Params: 26000 on epoch: 54]
    """
    def __init__(self):
        super(Simplenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu_conv1 = nn.ReLU()
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu_conv2 = nn.ReLU()
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool1(self.relu_conv1(self.conv1(x)))
        x = self.pool2(self.relu_conv2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.fc3(x)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def simplenet_cifar():
    model = Simplenet()
    return model