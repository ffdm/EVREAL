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
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import blosc2
from glob import glob
import os


# dataloader arguments
# FIXED AT 256 FOR BLOSC INPUTS
batch_size = 256
# 30 FPS using provided annotations
fps = 30

#data_path='../outputs/t50ms/eTraM_npy/train_h5_1/HyperE2VID/frame_0000001000.png'
#events_path='/data1/fdm/eTraM/Static/HDF5/train_h5_1/train_day_0001_td.h5'
#targets_path='../downstream_tasks/detection/outputs/HyperE2VID/train_h5_1/boxes/'

data_path = '/data1/fdm/eTraM/Static/HDF5/'
test_series = ['test_h5_1', 'test_h5_2']
train_series = ['train_h5_1', 'train_h5_2', 'train_h5_3', 'train_h5_4',
                'train_h5_5']

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
        # Output 16x176x316
        # MaxPool 16x88x158
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(16, 32, 5)
        # Output 32x84x154
        # MaxPool 32x42x77
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
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

def load_pretrained(file):
    net.load_state_dict(torch.load(file))
    net.eval()

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

    minibatches = 16
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
optimizer = torch.optim.Adam(net.parameters(), lr=2e-5, betas=(0.9, 0.999))

# loss = SF.mse_count_loss()
loss_fn = SF.ce_rate_loss()
# optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))

#train_data = to_frame(50).to(device)
#train_targets = get_targets(50).to(device)

# Epochs
num_scenes = 20
#num_epochs = 10
num_epochs = 1

# Batches
num_samples = 21
max_train = 16
#num_batches = 16
num_batches = 1

# Batches per epoch (batches per scene)
# Reserve 5 samples for test
loss_hist = []
test_acc_hist = []
counter = 0

# outer training loop
def train(iter_counter):
    ones = 0
    total = 0

    for series in train_series:
        if (series != 'train_h5_1'): break
        print(f" SERIES {series} ".center(50, "#"))
        path = data_path+series+'/'

        epoch_counter = 0

        for scenefile in glob(path+'*_td.h5'):
            scene = scenefile.replace(path, '').replace('_td.h5', '')
            
            if epoch_counter == num_epochs: break

            print(f" SCENE {scene} ".center(50, "#"))
            sys.stdout.flush()

            batch_counter = 0
            # Random sample batch from each scene
            batch_list = random.sample(range(max_train), num_batches)
            for batchfile in glob(path+'b2/'+scene+'_ev_b*.b2'):
                batch = batchfile.replace(path+'b2/', '').replace('.b2', '')
                batch = int(batch.replace(scene+'_ev_b', ''))

                if batch_counter == num_batches: break

                print(f" Batch {batch_counter} ".center(50, "-"))

                train_data = get_events(path, scene, batch)
                train_targets = get_targets(path, scene, batch).unsqueeze(1) 

                """
                # Shuffle within batch
                indexes = torch.randperm(train_data.shape[0])
                train_data = train_data[indexes]
                train_targets = train_targets[indexes]
                """
                
                # Count number of ones within batch
                ones += int(torch.sum(train_targets, dim=0))
                total += batch_size
                # print(f"Number of ones in batch: {int(ones)}/{batch_size}")
                # minibatch training loop
                minibatches = 16
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

                    # loss
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

            with torch.no_grad():
                net.eval()
                test_batch = random.sample(range(max_train, num_samples), 1)[0]
                test_acc = batch_accuracy(path, scene, test_batch, net)

                print(f"Test Acc: {test_acc * 100:.2f}%\n")
                test_acc_hist.append(test_acc.item())

    print(f"Total ones in training data: {ones}/{total}")

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
    num_test_scenes = 1
    num_batches = 1

    # FINAL ACCURACY MEASURE
    print("FINAL ACCURACY MEASURE")
    with torch.no_grad():
        total = 0
        acc = 0
        index = 0
        net.eval()
        for series in test_series:
            print(f" RUNNING SERIES {series} ".center(50, "#"))
            path = data_path+series+'/'

            test_scene_counter = 0

            for scenefile in glob(path+'*_td.h5'):
                scene = scenefile.replace(path, '').replace('_td.h5', '')

                if test_scene_counter == num_test_scenes: break

                print(f" RUNNING SCENE {scene} ".center(50, "#"))
                sys.stdout.flush()

                batch_counter = 0
                batch_list = random.sample(range(max_train, num_samples),4)

                for batchfile in glob(path+'b2/'+scene+'_ev_b*.b2'):
                    batch = batchfile.replace(path+'b2/', '').replace('.b2', '')
                    batch = int(batch.replace(scene+'_ev_b', ''))

                    if batch_counter == num_batches: break

                    print(f" Batch {batch_counter} ".center(50, "-"))

                    test_data = get_events(path, scene, batch)
                    test_targets = get_targets(path, scene, batch).unsqueeze(1) 

                    minibatches = 16
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
                test_scene_counter += 1
        print(f"Total correctly classified test set images: {correct}/{total}")
        print(f"Test set Accuracy: {100 * correct / total:.2f}%")

#t = time.process_time()
#train(counter)
#elapsed = time.process_time() - t
#print(f"Training loop took {elapsed} s")

# Save network weights
#torch.save(net.state_dict(), 'snn.pt')

load_pretrained('snn.pt')

t = time.process_time()
final_acc()
elapsed = time.process_time() - t

print(f"Final acc took {elapsed} s")

