# imports
import sys
import snntorch as snn
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch.functional import quant
from snntorch import surrogate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import cv2


# dataloader arguments
batch_size = 128
data_path=''

dtype = torch.float
device = torch.device("cuda")

# define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

#mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
#mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# create dataloaders
#train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
#test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Temporal Dynamics
num_steps = 25
beta = 0.95

spike_grad = surrogate.atan()

# Define Network
"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Init layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(12, 32, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc1 = nn.Linear(32*5*5, 2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):

        # Init hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            print(f"Input size: {x.size()}")
            #print(f"X: {x}")
            #cur1 = self.conv1(x[step])
            cur1 = self.conv1(x)
            print(f"conv1 Output size: {cur1.size()}")
            #print(f"conv1 Output: {cur1}")
            spk1, mem1 = self.lif1(cur1, mem1)
            print(f"lif1 Output size: {spk1.size()}")
            #print(f"spk1 : {spk1}")
            cur2 = self.conv2(spk1)
            print(f"conv2 Output size: {cur2.size()}")
            #print(f"conv2 Output: {cur2}")
            spk2, mem2 = self.lif2(cur2, mem2)
            print(f"lif2 Output size: {spk2.size()}")
            #print(f"lif2 Output: {spk2}")
            cur3 = self.fc1(spk2)
            #print(f"fc1 Output size: {cur3.size()}")
            #print(f"fc1 Output: {cur3}")
            spk3, mem3 = self.lif3(cur1, mem1)
            #print(f"lif3 Output size: {spk3.size()}")
            #print(f"spk3 : {spk3}")

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

"""

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc3(x)
        return x


net = Net().to(device)

image = cv2.imread('../outputs/t100ms/eTraM_npy/train_h5_1/HyperE2VID/frame_0000000000.png')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = torch.Tensor(image)
image = image.reshape((3, 720, 1280))
image = image.to(device)

print(image.shape)
print(type(image))
batch_size=1
#image = spikegen.rate(image.view(batch_size, -1), num_steps=num_steps)
#image = image.view(batch_size, -1)

print(image.shape)
# Run through network
spk_rec, mem_rec = net(image)
sys.exit()


# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(
        data, targets, epoch,
        counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

# Loss fn and optimizer

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

# loss = SF.mse_count_loss()
# optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))

# Take first batch of data
data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)
data = spikegen.rate(data.view(batch_size, -1), num_steps=num_steps)


# run through neural network
spk_rec, mem_rec = net(data)

num_epochs = 2
loss_hist = []
test_loss_hist = []
counter = 0

# outer training loop
def train(counter):
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)

        # minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)
            
            # convert data to spike train
            data = spikegen.rate(data.view(batch_size, -1), num_steps=num_steps)

            # forward
            net.train()
            spk_rec, mem_rec = net(data)

            # init loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # grad calc + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            # store loss history for plotting
            loss_hist.append(loss_val.item())

            # test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)
                test_data = spikegen.rate(test_data.view(batch_size, -1), num_steps=num_steps)

                # Test set forward pass
                test_spk, test_mem = net(test_data)

                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # print train/test loss accuracy
                if counter % 50 == 0:
                    train_printer(
                            data, targets, epoch,
                            counter, iter_counter,
                            loss_hist, test_loss_hist,
                            test_data, test_targets)
                counter += 1
                iter_counter += 1

            
    # plot loss over iteration
    fig, ax = plt.subplots()
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.legend(["train", "test"])
    plt.ylabel("loss")
    plt.xlabel("iter")
    plt.savefig("loss.pdf")

    total=0
    correct=0

    # drop_last switch to False to keep all samples
    final_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

    with torch.no_grad():
        net.eval()
        for data, targets in final_test_loader:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            test_spk, _ = net(spikegen.rate(data.view(data.size(0), -1),
                num_steps=num_steps))

            # calc total acc
            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test set Accuracy: {100 * correct / total:.2f}%")

    # Save network weights
    torch.save(net.state_dict(), 'easy2.net')

train(counter)
