### Copyright 2023 [Dawn Of Eve]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os, random
from typing import Any, List, Tuple
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nadir import nadir as optim


from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, utils

import wandb
from tqdm import tqdm

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True

# Make a Namespace object to store all the experiment values
args = argparse.Namespace()

args.learning_rate : float = 1e-3
args.batch_size : int = 64
args.test_batch_size : int = 1000
args.gamma : float = 0.7
args.device : bool = 'cuda' if torch.cuda.is_available() else 'cpu'
args.log_interval : int = 10
args.epochs : int = 10
args.betas : Tuple[float, float] = (0.9, 0.99)
args.eps : float = 1e-16
args.optimizer : Any = optim.Adam

# with open("random_seeds.txt", 'r') as file:
#     file_str = file.read().split('\n')
#     seeds = [int(num) for num in file_str]
args.random_seeds : List[int] = [42]

args.seed : int = args.random_seeds[0]

# writing the logging args as a namespace obj
largs = argparse.Namespace()
largs.run_name : str = 'DoE-Adam'
largs.run_seed : str = args.seed


# Initialising the seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

class MNISTestNet(nn.Module):
    def __init__(self):
        super(MNISTestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
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
        output = x
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for (data, target)in (pbar := tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item() : .5f}")
        wandb.log({'train/Loss': loss.item()})
    return loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in (pbar := tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='mean').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()

            pbar.set_description(f"Accuracy: {correct/len(test_loader.dataset) : .4f}")
    test_loss /= len(test_loader.dataset)
    wandb.log({'test/Accuracy': correct/len(test_loader.dataset)})
    wandb.log({'test/Loss': test_loss})
    return test_loss


def prepare_loaders(args, use_cuda=False):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    def seed_worker():
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../../data',
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn = seed_worker,
        generator=generator,
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
        batch_size=args.test_batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator,
        **kwargs,
    )
    return train_loader, test_loader



def mnist_tester(optimizer=None, args = None, largs = None):
    train_loss = []
    test_loss = []
    
    torch.manual_seed(args.random_seeds[0])
    device = args.device
    use_cuda = True if device == torch.device('cuda') else False
    train_loader, test_loader = prepare_loaders(args, use_cuda)

    model = MNISTestNet().to(device)

    # create grid of images and write to wandb
    images, labels = next(iter(train_loader))
    img_grid = utils.make_grid(images)
    wandb.log({'mnist_images': img_grid})

    # custom optimizer from torch_optimizer package
    if args.optimizer == optim.SGD:
        config = optim.SGDConfig(lr=args.learning_rate)
    elif args.optimizer == optim.Adam:
        config = optim.AdamConfig(lr=args.learning_rate, betas=args.betas, eps=args.eps)
    # config = config(lr=args.learning_rate)
    optimizer = optimizer(model.parameters(), config)
    # optimizer = optim(model.parameters(), lr=args.learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)



    for epoch in (pbar := tqdm(range(1,  args.epochs + 1))):
        loss=train(args, model, device, train_loader, optimizer, epoch)
        tloss=test(model, device, test_loader)
        scheduler.step()
        train_loss.append(loss)
        test_loss.append(tloss)
        pbar.set_description(f"Loss: {loss:.5f}")
    return train_loss, test_loss


# If the file is run directly, follow the following behaviour
if __name__ == '__main__' :

    # Initialising the WANDB project
    run = wandb.init(project="MNIST", entity="dawn-of-eve")
    run.name = f'{largs.run_name}'
    run.config.update(args)
    run.config.update(largs)

    

    # Initialising the optimiser
    optimizer = args.optimizer
    #    config = AutoConfig(args.params..)
    #    optimizer = args.optimizer(config)


    # Running the mnist_tester
    mnist_tester(optimizer, args, largs)

    run.finish()