# imports
import sys
import snntorch as snn
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch.functional import quant
from snntorch import surrogate
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import blosc2


# dataloader arguments
# FIXED AT 256 FOR BLOSC INPUTS
batch_size = 256
# 30 FPS using provided annotations
fps = 30

#data_path='../outputs/t50ms/eTraM_npy/train_h5_1/HyperE2VID/frame_0000001000.png'
#events_path='/data1/fdm/eTraM/Static/HDF5/train_h5_1/train_day_0001_td.h5'
#targets_path='../downstream_tasks/detection/outputs/HyperE2VID/train_h5_1/boxes/'

data_path = '/data1/fdm/eTraM/Static/HDF5/'
series_path = 'train_h5_1/'

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
        """
        self.conv1 = nn.Conv2d(2, 1, 65)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc1 = nn.Linear(624, 128)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(128, 2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        """
        # Input 2x180x320
        self.conv1 = nn.Conv2d(2, 16, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # Output 16x176x316
        # MaxPool 16x88x158
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # Output 32x84x154
        # MaxPool 32x42x77
        self.fc1 = nn.Linear(32*42*77,2)
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
            """
            cur1 = F.max_pool2d(x[:, step], 8)
            cur1 = F.max_pool2d(self.conv1(cur1),2)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = torch.flatten(spk1, 1)
            cur2 = self.fc1(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc2(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            """
            # Do the maxpool before storing the data to decrease datasize
            cur1 = F.max_pool2d(x[:, step], 4)
            cur1 = F.max_pool2d(self.conv1(cur1),2)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = F.max_pool2d(self.conv2(spk1),2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = torch.flatten(spk2, 1)

            cur3 = self.fc1(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

net = Net().to(device)

# Goal: create represention of events as follows:
# Want to divide up the number of time bins between each frame
# Frames are defined by FPS, time bins defined by num_steps
# dim should be [N, C, H, W] = [#frames, 2, 720, 1280]

#print(x.shape)


# List of objects of interest
objects = ['pedestrian', 'car', 'bicycle', 'bus', 'motorbike', 'truck',
           'tram', 'wheelchair']

def get_events(path, prefix, bn):
    events = blosc2.load_array(path+'b2/'+prefix+'_ev_b'+str(bn).zfill(2)+'.b2')
    return torch.Tensor(events)

def get_targets(path, prefix, bn):
    targets = blosc2.load_array(path+'b2/'+prefix+'_tg_b'+str(bn).zfill(2)+'.b2')
    return torch.Tensor(targets)

path = data_path+series_path
prefix = 'train_day_0001'

# 10 train, 5 test batches for now

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def batch_accuracy(path, prefix, bn, net):
  with torch.no_grad():
    total = 0
    acc = 0
    index = 0
    net.eval()

    test_data = get_events(path, prefix, bn)
    test_targets = get_targets(path, prefix, bn).unsqueeze(1) 

    minibatches = 4
    # number of iters per minibatch
    num = int(len(test_data)/minibatches)
    for minibatch in range(minibatches):
        start = minibatch*num
        end = start + num
        data = test_data[start:end].to(device)
        targets = test_targets[start:end].type(torch.LongTensor)
        targets = targets.to(device).squeeze(1)

        spk_rec, _ = net(data)

        acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
        total += spk_rec.size(1)

  return acc/total

# Loss fn and optimizer

#loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))

# loss = SF.mse_count_loss()
loss_fn = SF.ce_rate_loss()
# optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))

#train_data = to_frame(50).to(device)
#train_targets = get_targets(50).to(device)

num_scenes = 20
num_samples = 21

max_train = 16

num_epochs = 20
num_batches = 5

# Batches per epoch (batches per scene)
# Reserve 5 samples for test
loss_hist = []
test_acc_hist = []
counter = 0

# outer training loop
def train(iter_counter):
    epoch_list = random.sample(range(num_scenes), num_epochs)
    epoch_counter = 0

    for epoch in epoch_list: 
        print(f" EPOCH {epoch_counter} ".center(50, "#"))
        # Flush output buffer
        sys.stdout.flush()
        batch_counter = 0
        # Random sample batch from each scene
        batch_list = random.sample(range(max_train), num_batches)
        for batch in batch_list:
            print(f" Batch {batch_counter} ".center(50, "-"))
            prefix = f"train_day_{str(epoch+1).zfill(4)}"

            train_data = get_events(path, prefix, batch)
            train_targets = get_targets(path, prefix, batch).unsqueeze(1) 

            """
            # Shuffle within batch
            indexes = torch.randperm(train_data.shape[0])
            train_data = train_data[indexes]
            train_targets = train_targets[indexes]
            """
            
            # Count number of ones within batch
            # ones = torch.sum(train_targets, dim=0)
            # print(f"Number of ones in batch: {int(ones)}/{batch_size}")

            # minibatch training loop
            # how much you divide batch by
            minibatches = 8
            # number of iters per minibatch
            num = int(len(train_data)/minibatches)

            for minibatch in range(minibatches):
                start = minibatch*num
                end = start + num
                data = train_data[start:end].to(device)
                targets = train_targets[start:end].type(torch.LongTensor)
                targets = targets.to(device).squeeze(1)

                # forward
                net.train()
                spk_rec, mem_rec = net(data)

                # init loss & sum over time
                #loss_val = torch.zeros((1), dtype=dtype, device=device)
                #for step in range(num_steps):
                #    loss_val += loss(mem_rec[step], targets)

                loss_val = loss_fn(spk_rec, targets)
                _, idx = spk_rec.sum(dim=0).max(1)

                # grad calc + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
        
                # store loss history for plotting
                loss_hist.append(loss_val.item())

                iter_counter += 1

            batch_counter += 1

        epoch_counter += 1
        test_batch = random.sample(range(max_train, num_samples), 1)[0]
        test_acc = batch_accuracy(path, prefix, test_batch, net)
        print(f"Test Acc: {test_acc * 100:.2f}%\n")
        test_acc_hist.append(test_acc.item())

    # plot loss over iteration
    fig, ax = plt.subplots()
    plt.plot(loss_hist)
    plt.legend("train loss")
    plt.xlabel("iter")
    plt.savefig("loss.pdf")
    plt.clf()
    plt.plot(test_acc_hist)
    plt.legend("test accuracy")
    plt.xlabel("iter")
    plt.savefig("acc.pdf")

def final_acc():

    total=0
    correct=0

    # FINAL ACCURACY MEASURE
    print("FINAL ACCURACY MEASURE")
    with torch.no_grad():
        total = 0
        acc = 0
        index = 0
        net.eval()
        scene_list = random.sample(range(num_epochs),4)
        for scene in scene_list:
            print(f" SCENE {scene} ".center(50, "#"))
            prefix = f"train_day_{str(scene+1).zfill(4)}"

            batch_list = random.sample(range(num_samples),4)
            batch_counter = 0
            for batch in batch_list:
                print(f" Batch {batch_counter} ".center(50, "-"))
                test_data = get_events(path, prefix, batch)
                test_targets = get_targets(path, prefix, batch).unsqueeze(1) 
                
                minibatches = 8
                # number of iters per minibatch
                num = int(len(test_data)/minibatches)
                for minibatch in range(minibatches):
                    start = minibatch*num
                    end = start + num
                    data = test_data[start:end].to(device)
                    targets = test_targets[start:end].type(torch.LongTensor)
                    targets = targets.to(device).squeeze(1)

                    spk_rec, _ = net(data)

                    acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                    _, predicted = spk_rec.sum(dim=0).max(1)

                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                batch_counter += 1

        print(f"Total correctly classified test set images: {correct}/{total}")
        print(f"Test set Accuracy: {100 * correct / total:.2f}%")


train(counter)
final_acc()
# Save network weights
#torch.save(net.state_dict(), 'snn.net')

