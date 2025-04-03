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
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm


# dataloader arguments
batch_size = 256

data_path='../outputs/t50ms/eTraM_npy/train_h5_1/HyperE2VID/frame_0000001000.png'
events_path='/data1/fdm/eTraM/Static/HDF5/train_h5_1/train_day_0001_td.h5'
targets_path='../downstream_tasks/detection/outputs/HyperE2VID/train_h5_1/boxes/'

dtype = torch.float

# CUDA for on MBIT
device = torch.device("cuda")
#device = torch.device("cpu")

# Temporal Dynamics
num_steps = 25
beta = 0.95

spike_grad = surrogate.atan()

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 1, 65)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc1 = nn.Linear(624, 128)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(128, 2)
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
            # Max pool of 8
            cur1 = F.max_pool2d(x[:, step], 8)
            cur1 = F.max_pool2d(self.conv1(cur1),2)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = torch.flatten(spk1, 1)
            cur2 = self.fc1(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc2(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

net = Net().to(device)

#image = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
#image = torch.from_numpy(image)
#image = image.to(device)
#image = image.unsqueeze(0) / 255.0

#image = spikegen.rate(image, num_steps=num_steps)
#print(image.shape)

import h5py
events = h5py.File(events_path, 'r')['events']

x = events['x'][()]
y = events['y'][()]
h = events['height'][()]
w = events['width'][()]
p = events['p'][()]
t = events['t'][()]

# Goal: create represention of events as follows:
# Want to divide up the number of time bins between each frame
# Frames are defined by FPS, time bins defined by num_steps
# dim should be [N, C, H, W] = [#frames, 2, 720, 1280]

print(x.shape)

fps = 20
# List of objects of interest
objects = ['person', 'car']

# If wanted just the frame of events. Instead, we just need to calculate this
# Same type of thing but then do it for finer grained increments
def to_frame(beg, end):
    ind = 0
    ind_p = 0
    print(f" LOADING {end-beg} DATA VECTORS ".center(50, "#"))
    data = torch.zeros((end-beg, num_steps, 2, h, w))
    tn = 0
    for i in tqdm(range(beg, end)):
        # Can be like 600k events in one frame
        for n in range(num_steps):
            # Max time in us for bin n
            tn += 1e6/fps/num_steps
            ind_p = ind
            while(t[ind] < tn):
                ind += 1
            x_bin = x[ind_p:ind+1]
            y_bin = y[ind_p:ind+1]
            p_bin = p[ind_p:ind+1]
            data[i-beg, n, p_bin, y_bin, x_bin] = 1

    return data

def get_targets(beg, end):
    targets = torch.zeros(end - beg)
    print(f" LOADING {end-beg} TARGETS ".center(50, "#"))
    for n in tqdm(range(beg, end)):
        f = open(targets_path+f'frame_{str(n).zfill(5)}.txt', 'r')
        target = 0
        for line in f:
            name = line.split()[0]
            if (name in objects): 
                target = 1
                break
        f.close()
        targets[n-beg] = target
    return targets

def get_data(beg, end):
    return to_frame(beg, end), get_targets(beg, end)


# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def batch_accuracy(beg, end, net):
  with torch.no_grad():
    total = 0
    acc = 0
    index = 0
    net.eval()

    test_data = to_frame(beg, end)
    test_targets = get_targets(beg, end).unsqueeze(1) 

    for j, data in enumerate(test_data):
      data = data.unsqueeze(0).to(device)
      targets = test_targets[j].type(torch.LongTensor)
      targets = targets.to(device)
      spk_rec, _ = net(data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      _, idx = spk_rec.sum(dim=0).max(1)
      index += idx
      total += spk_rec.size(1)

  print(f"Average Predicted Class {index/total}")
  return acc/total
# Loss fn and optimizer

#loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(0.4, 0.9))

# loss = SF.mse_count_loss()
loss_fn = SF.ce_rate_loss()
# optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))

#train_data = to_frame(50).to(device)
#train_targets = get_targets(50).to(device)

train_beg = 1000
train_end = 3000

test_beg = 3000
test_end = 3597

num_epochs = 3
loss_hist = []
test_acc_hist = []
counter = 0

# outer training loop
def train(counter):
    for epoch in range(num_epochs):
        #train_batch = iter(train_loader)
        batch_beg = train_beg+counter
        batch_end = batch_beg+batch_size
        train_data = to_frame(batch_beg, batch_end)
        train_targets = get_targets(batch_beg, batch_end).unsqueeze(1) 

        indexes = torch.randperm(train_data.shape[0])
        train_data = train_data[indexes]
        train_targets = train_targets[indexes]

        # minibatch training loop
        #for data, targets in train_batch:
        print(f" EPOCH {epoch} ".center(50, "#"))
        for j, data in enumerate(train_data):
            data = data.unsqueeze(0).to(device)
            targets = train_targets[j].type(torch.LongTensor)
            targets = targets.to(device)
            
            # convert data to spike train
            #data = spikegen.rate(data.view(batch_size, -1), num_steps=num_steps)

            # forward
            net.train()
            spk_rec, mem_rec = net(data)

            # init loss & sum over time
            #loss_val = torch.zeros((1), dtype=dtype, device=device)
            #for step in range(num_steps):
            #    loss_val += loss(mem_rec[step], targets)

            loss_val = loss_fn(spk_rec, targets)

            # grad calc + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            # store loss history for plotting
            loss_hist.append(loss_val.item())

            # test set
            if counter % 50 == 0:
                with torch.no_grad():
                    net.eval()
                    # Test set forward pass
                    # Batch is not random or changing as of now
                    test_acc = batch_accuracy(test_beg, test_beg+batch_size, net)
                    print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())
            counter += 1

            
    # plot loss over iteration
    fig, ax = plt.subplots()
    plt.plot(loss_hist)
    plt.plot(test_acc_hist)
    plt.legend(["train loss", "test accuracy"])
    plt.xlabel("iter")
    plt.savefig("loss.pdf")

    sys.exit()

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
