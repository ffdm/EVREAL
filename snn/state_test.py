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

import functools
print = functools.partial(print, flush=True)


# dataloader arguments
batch_size = 128
# 30 FPS using provided annotations
fps = 30

data_path = '/data1/fdm/eTraM/Static/HDF5/'
test_series = ['test_h5_1', 'test_h5_2']
train_series = ['train_h5_1', 'train_h5_2', 'train_h5_3', 'train_h5_4',
                'train_h5_5']

# List of objects of interest
objects = ['pedestrian', 'car', 'bicycle', 'bus', 'motorbike', 'truck',
           'tram', 'wheelchair']

# CUDA for on MBIT
device = torch.device("cuda")

# Temporal Dynamics
num_steps = 25
beta = 0.95

spike_grad = surrogate.atan()

# Global vars for target gen
total_iou = 0
total_seen = 0
dropped = 0
min_seen_iou = 1

# IOU threshold to send
min_iou = 1

mem1 = self.lif1.init_leaky()
mem2 = self.lif2.init_leaky()
mem3 = self.lif3.init_leaky()

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

import numpy as np

def get_events(path, prefix, bn):
    events = blosc2.load_array(path+'b2/'+prefix+'_ev_b'+str(bn).zfill(2)+'.b2')
    return torch.Tensor(events)

def iterate_through():
    for series in test_series:
        path = data_path+series+'/'

        test_scene_counter = 0

        print(f"Series: {series}")

        for scenefile in glob(path+'*_td.h5'):
            scene = scenefile.replace(path, '').replace('_td.h5', '')

            batch_counter = 0

            for batchfile in glob(path+'b2/'+scene+'_tg_b*.npy'):
                batch = batchfile.replace(path+'b2/', '').replace('.npy', '')
                batch = int(batch.replace(scene+'_tg_b', ''))

                test_data = get_events(path, scene, batch)

                batch_counter += 1
            test_scene_counter += 1

        print(f"Avg IOU: {total_iou/total_seen}, lowest IOU: {min_seen_iou}")
        print(f"dropped: {dropped}/{total_seen}={dropped/total_seen:.4f}")

# Loss fn and optimizer

optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(0.8, 0.95))
#optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))

loss_fn = SF.ce_rate_loss()

# Epochs
num_epochs = 5

# Batches
num_samples = 21
max_train = 16
num_batches = 10

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
        print(f" SERIES {series} ".center(50, "#"))
        path = data_path+series+'/'

        epoch_counter = 0
        
        scenefiles = glob(path+'*_td.h5')
        random.shuffle(scenefiles)
        for scenefile in scenefiles:
            #if epoch_counter == num_epochs: break

            scene = scenefile.replace(path, '').replace('_td.h5', '')
            print(f" SCENE {scene} ".center(50, "#"))

            batch_counter = 0

            batchfiles = glob(path+'b2/'+scene+'_tg_b*.npy')
            random.shuffle(batchfiles)
            for batchfile in batchfiles:
                #if batch_counter == num_batches: break

                batch = batchfile.replace(path+'b2/', '').replace('.npy', '')
                batch = int(batch.replace(scene+'_tg_b', ''))
                print(f" Batch {batch_counter} ({batch}) ".center(50, "-"))

                train_data = get_events(path, scene, batch)
                train_targets = get_targets(path, scene, batch).unsqueeze(1) 

                """
                # Shuffle within batch
                indexes = torch.randperm(train_data.shape[0])
                train_data = train_data[indexes]
                train_targets = train_targets[indexes]
                """
                
                total += batch_size

                # minibatch training loop
                minibatches = 4
                # number of iters per minibatch
                num = int(len(train_data)/minibatches)

                for minibatch in range(minibatches):
                    start = minibatch*num
                    end = start + num
                    data = train_data[start:end].to(device)
                    targets = train_targets[start:end].type(torch.LongTensor)
                    targets = targets.to(device).squeeze(1)

                    # Count number of ones within minibatch
                    mb_ones = int(torch.sum(targets, dim=0))
                    """
                    if mb_ones == num:
                        print("Skipping minibatch (all ones)")
                        continue
                    elif mb_ones == 0:
                        print("Skipping minibatch (all zeros)")
                        continue
                    """

                    ones += mb_ones
                    print(f"Percent of ones in minibatch: {100*mb_ones/num:.2f}%")

                    # forward
                    net.train()
                    spk_rec, mem_rec = net(data)

                    # loss
                    loss_val = loss_fn(spk_rec, targets)
                    _, idx = spk_rec.sum(dim=0).max(1)

                    print(spk_rec.sum(dim=0)[0])
                    print(targets[0])

                    """
                    log_softmax_fn = nn.LogSoftmax(dim=-1)
                    loss_fn_nll = nn.NLLLoss()
                    log_p_y = log_softmax_fn(spk_rec)
                    print(f"log_p_y: {log_p_y}")
                    print(f"log_p_y shape: {log_p_y.shape}")
                    print(f"output fn nll: {loss_fn_nll(log_p_y[0], targets)}")
                    print(f"output fn nll shape: {loss_fn_nll(log_p_y[0],targets).shape}")

                    loss = torch.zeros(1, device=device)
                    for step in range(num_steps):
                        loss += loss_fn_nll(log_p_y[step], targets)

                    print(f"avg loss over num steps: {loss/num_steps}")
                    """

                    # grad calc + weight update
                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()
            
                    # store loss history for plotting
                    loss_hist.append(loss_val.item())

                    # minibatch accuracy (before weight update)
                    acc = np.mean((targets == idx).detach().cpu().numpy())
                    test_acc_hist.append(acc)
                    print(f"Minibatch accuracy: {100*acc:.2f}%")
                    print(f"Minibatch loss: {loss_val.item():.2f}")

                    iter_counter += 1

                batch_counter += 1

            epoch_counter += 1

    print(f"Total ones in training data: {ones}/{total}")

    # plot loss over iteration
    fig, ax = plt.subplots()
    plt.plot(loss_hist)
    plt.xlabel("iter")
    plt.ylabel("minibatch loss")
    plt.savefig("loss.pdf")
    plt.clf()
    plt.plot(test_acc_hist)
    plt.xlabel("iter")
    plt.ylabel("minibatch acc")
    plt.savefig("acc.pdf")

def final_acc():

    total=0
    correct=0

    # set to 99 to do all of them
    num_test_scenes = 5
    num_batches = 5

    # FINAL ACCURACY MEASURE
    print("FINAL ACCURACY MEASURE")
    with torch.no_grad():
        total = 0
        acc = 0
        index = 0
        ones = 0
        net.eval()
        for series in test_series:
            print(f" RUNNING SERIES {series} ".center(50, "#"))
            path = data_path+series+'/'

            test_scene_counter = 0

            scenefiles = glob(path+'*_td.h5')
            random.shuffle(scenefiles)
            for scenefile in scenefiles:
                scene = scenefile.replace(path, '').replace('_td.h5', '')

                #if test_scene_counter == num_test_scenes: break

                print(f" RUNNING SCENE {scene} ".center(50, "#"))

                batch_counter = 0

                batchfiles = glob(path+'b2/'+scene+'_tg_b*.npy')
                random.shuffle(batchfiles)
                for batchfile in batchfiles:
                    #if batch_counter == num_batches: break

                    batch = batchfile.replace(path+'b2/', '').replace('.npy', '')
                    batch = int(batch.replace(scene+'_tg_b', ''))
                    print(f" Batch {batch_counter} ".center(50, "-"))

                    test_data = get_events(path, scene, batch)
                    test_targets = get_targets(path, scene, batch).unsqueeze(1) 

                    ones += int(torch.sum(test_targets, dim=0))

                    minibatches = 2
                    # number of iters per minibatch
                    num = int(len(test_data)/minibatches)

                    for minibatch in range(minibatches):
                        start = minibatch*num
                        end = start + num
                        data = test_data[start:end].to(device)
                        targets = test_targets[start:end].type(torch.LongTensor)
                        targets = targets.to(device).squeeze(1)

                        spk_rec, _ = net(data)
                        _, predicted = spk_rec.sum(dim=0).max(1)

                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                    batch_counter += 1
                test_scene_counter += 1

        print(f"Total correctly classified test set images: {correct}/{total}")
        print(f"Test set Accuracy: {100 * correct / total:.2f}%")
        print(f"Total ones in test set: {ones}")

start = time.time()
train(counter)
#iterate_through()
end = time.time()
print(f"Training loop took {end-start:.2f} s")

# Save network weights
torch.save(net.state_dict(), 'snn.pt')
start = time.time()
final_acc()
end = time.time()
print(f"Final acc took {end-start:.2f} s")

#load_pretrained('snn.pt')
