import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import wandb
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from torch import optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(conf, model, device, train_loader, optimizer, epoch, wandb):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % conf.log_interval == 0:
            loss = loss.item()
            idx = batch_idx + epoch * (len(train_loader))
            wandb.log({'Loss/train': loss})
            # print(
            #     'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch,
            #         batch_idx * len(data),
            #         len(train_loader.dataset),
            #         100.0 * batch_idx / len(train_loader),
            #         loss,
            #     )
            # )
    return loss


def test(conf, model, device, test_loader, epoch, wandb):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                                                                            test_loss, 
                                                                            correct, 
                                                                            len(test_loader.dataset),
                                                                            100. * correct / len(test_loader.dataset)
                                                                            ))

    wandb.log({'Accuracy': correct})
    wandb.log({'Loss/test': test_loss})
    return test_loss


def prepare_loaders(conf, use_cuda=False):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=conf.batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=conf.test_batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


class Config:
    def __init__(
        self,
        batch_size: int = 64,
        test_batch_size: int = 1000,
        epochs: int = 5,
        lr: float = 0.01,
        gamma: float = 0.7,
        no_cuda: bool = True,
        seed: int = 42,
        log_interval: int = 10,
    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.no_cuda = no_cuda
        self.seed = seed
        self.log_interval = log_interval


def MNIST_tester(optim=None):
    conf = Config()
    train_loss = []
    test_loss = []
    log_dir = 'runs/mnist_custom_optim'
    wandb.init(project="test-project", entity="dawn-of-eve")
    wandb.config = {
        "learning_rate": conf.lr,
        "epochs": conf.epochs,
        "batch_size": conf.batch_size,
        "gamma": conf.gamma
    }
    use_cuda = not conf.no_cuda and torch.cuda.is_available()
    torch.manual_seed(conf.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_loader, test_loader = prepare_loaders(conf, use_cuda)

    model = Net().to(device)

    # create grid of images and write to wandb
    images, labels = next(iter(train_loader))
    img_grid = utils.make_grid(images)
    wandb.log({'mnist_images': img_grid})

    # custom optimizer from torch_optimizer package
    optimizer = optim(model.parameters(), lr=conf.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=conf.gamma)
    for epoch in tqdm(range(1, conf.epochs + 1)):
        loss=train(conf, model, device, train_loader, optimizer, epoch, wandb)
        tloss=test(conf, model, device, test_loader, epoch, wandb)
        scheduler.step()
        train_loss.append(loss)
        test_loss.append(tloss)
        print(
            'Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch,
            loss
    )
)
    return train_loss, test_loss




