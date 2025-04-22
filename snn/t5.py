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
batch_size = 128
# 30 FPS using provided annotations
fps = 30

#data_path='../outputs/t50ms/eTraM_npy/train_h5_1/HyperE2VID/frame_0000001000.png'
#events_path='/data1/fdm/eTraM/Static/HDF5/train_h5_1/train_day_0001_td.h5'
#targets_path='../downstream_tasks/detection/outputs/HyperE2VID/train_h5_1/boxes/'

data_path = '/data1/fdm/eTraM/Static/HDF5/'
test_series = ['test_h5_1', 'test_h5_2']
train_series = ['train_h5_1', 'train_h5_2', 'train_h5_3', 'train_h5_4',
                'train_h5_5']

# List of objects of interest
objects = ['pedestrian', 'car', 'bicycle', 'bus', 'motorbike', 'truck',
           'tram', 'wheelchair']

bb_delta = 5

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

#########################################
# FUNCTIONS FOR DISPLAYING EVENTS AND BBs
#########################################

def rei_frame(image, events):
    for t in tqdm(range(events.shape[0])):
        image[np.where(events[t,0] == 1)] = np.array([0,0,255])
        image[np.where(events[t,1] == 1)] = np.array([255,0,0])
    return image

def draw_detection(image, bbox, label):
    """
    Draw bounding box and label on the image.
    """
    bbox = list(map(int, bbox))
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image, f"{label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def render(events, targets):
    for i, event_frame in enumerate(events):
        render_frame(event_frame, targets[i])

def render_frame(events, targets):
    black_image = np.zeros((events.shape[2], events.shape[3], 3))
    frame = rei_frame(black_image, events)

    for ann in targets:
        x1 = ann['x']
        y1 = ann['y']
        x2 = x1 + ann['w']
        y2 = y1 + ann['h']
        label = f"{ann['class_confidence']:.2f}: {objects[ann['class_id']]}"
        draw_detection(frame, (x1, y1, x2, y2), label)

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

#########################################
# FUNCTIONS FOR GETTING EVENTS AND TARGETS
#########################################

import scipy.optimize
import numpy as np

def bbox_iou(boxA, boxB):
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.

  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou


def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2].
      The number of bboxes, N1 and N2, need not be the same.

    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate( (iou_matrix,
                                    np.full((diff, n_pred), MIN_IOU)),
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate( (iou_matrix,
                                    np.full((n_true, diff), MIN_IOU)),
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label

def get_events(path, prefix, bn):
    events = blosc2.load_array(path+'b2/'+prefix+'_ev_b'+str(bn).zfill(2)+'.b2')
    return torch.Tensor(events)

# MOVE THIS IN A BIT
total_iou = 0
total_seen = 0
dropped = 0
min_seen_iou = 1
min_iou = 0.90

def get_targets(path, prefix, bn):
    global total_iou
    global total_seen
    global dropped
    global min_seen_iou
    targets = np.zeros(batch_size)

    anns = np.load(path+'b2/'+prefix+'_tg_b'+str(bn).zfill(2)+'.npy',
                    allow_pickle=True)

    # render_frame(get_events(path, prefix, bn)[15], anns[15])

    # TARGET GENERATION METHODOLOGY
    # FIRST IMAGE ALWAYS IS MARKED AS A 1 to encourage SNN to send more
    # often than not. The target is then saved. 
    # If the number of BBs changes? SNN sends
    # Else, the iou is calciulated from previous send
    # If it is below a threshold? the SNN sends

    # ANNOTATION FOR PREVIOUSLY DETECTED FRAME
    ann_p = np.array([-1.0,0,0,0,0,0,0,0], dtype=anns[0].dtype)

    for i, ann in enumerate(anns):
        if len(ann) == 0: continue

        total_seen += 1
        #print("".center(50,'#'))
        #print(f"X: {ann['x']}, Y: {ann['y']}")
        #print(f"PX: {ann_p['x']}, PY: {ann_p['y']}")

        #print(ann_p['x'])
        #print(ann['x'])
        if len(ann_p) != len(ann) or ann_p['x'][0] == -1:
            targets[i] = 1
            ann_p = ann
            total_iou += 1
            #print("IOU: 1")
            continue
        
        #xdiff = np.max(np.abs(ann['x']-ann_p['x']))
        #ydiff = np.max(np.abs(ann['y']-ann_p['y']))

        #diff = xdiff+ydiff

        x1 = ann['x']
        y1 = ann['y']
        x2 = x1 + ann['w']
        y2 = y1 + ann['h']
        bbox = np.column_stack((x1, y1, x2, y2))

        x1_p = ann_p['x']
        y1_p = ann_p['y']
        x2_p = x1_p + ann_p['w']
        y2_p = y1_p + ann_p['h']
        bbox_p = np.column_stack((x1_p, y1_p, x2_p, y2_p))
        
        (idxs_true, idxs_pred, ious, labels) = match_bboxes(bbox, bbox_p) 
        #print(ious)
        if len(ious) > 0: iou_min = np.min(ious)
        else: iou_min = min_seen_iou

        if iou_min < min_iou:
            targets[i] = 1
            ann_p = ann
            total_iou += 1
            #print("IOU: 1")
        else:
            total_iou += iou_min
            dropped += 1
            #print(f"IOU: {iou_min}")
            if (min_seen_iou > iou_min):
                min_seen_iou = iou_min

        """
        if diff > bb_delta:
            targets[i] = 1
            ann_p = ann
        """
    
    return torch.Tensor(targets)

# 10 train, 5 test batches for now

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def iterate_through():
    for series in test_series:
        path = data_path+series+'/'

        test_scene_counter = 0

        for scenefile in glob(path+'*_td.h5'):
            scene = scenefile.replace(path, '').replace('_td.h5', '')

            batch_counter = 0

            for batchfile in glob(path+'b2/'+scene+'_ev_b*.b2'):
                batch = batchfile.replace(path+'b2/', '').replace('.b2', '')
                batch = int(batch.replace(scene+'_ev_b', ''))

                #test_data = get_events(path, scene, batch)
                test_targets = get_targets(path, scene, batch).unsqueeze(1) 
                #test_targets = get_targets(path, scene, batch)
                print(f"Avg IOU: {total_iou/total_seen}, lowest IOU: {min_seen_iou}")
                print(f"dropped: {dropped}/{total_seen}={dropped/total_seen:.4f}")

                batch_counter += 1
            test_scene_counter += 1





def batch_accuracy():
    total=0
    correct=0
    # set to 99 to do all of them
    num_test_scenes = 2
    num_batches = 2

    with torch.no_grad():
        total = 0
        acc = 0
        index = 0
        net.eval()
        for series in test_series:

            path = data_path+series+'/'

            test_scene_counter = 0

            for scenefile in glob(path+'*_td.h5'):
                scene = scenefile.replace(path, '').replace('_td.h5', '')

                if test_scene_counter == num_test_scenes: break

                batch_counter = 0

                for batchfile in glob(path+'b2/'+scene+'_ev_b*.b2'):
                    batch = batchfile.replace(path+'b2/', '').replace('.b2', '')
                    batch = int(batch.replace(scene+'_ev_b', ''))

                    if batch_counter == num_batches: break

                    test_data = get_events(path, scene, batch)
                    test_targets = get_targets(path, scene, batch).unsqueeze(1) 

                    minibatches = 64
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

        return correct/total

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
num_epochs = 10

# Batches
num_samples = 21
max_train = 16
#num_batches = 16
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
        # if (series != 'train_h5_1'): break
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
                sys.stdout.flush()

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

            """
            with torch.no_grad():
                net.eval()
                test_batch = random.sample(range(max_train, num_samples), 1)[0]
                print(f"Running batch accuracy")
                sys.stdout.flush()
                test_acc = batch_accuracy()

                print(f"Test Acc: {test_acc * 100:.2f}%\n")
                sys.stdout.flush()
                test_acc_hist.append(test_acc.item())
            """

    print(f"Total ones in training data: {ones}/{total}")

    # plot loss over iteration
    fig, ax = plt.subplots()
    plt.plot(loss_hist)
    plt.legend("train loss")
    plt.xlabel("iter")
    plt.savefig("loss.pdf")
    plt.clf()
    """
    plt.plot(test_acc_hist)
    plt.legend("test accuracy")
    plt.xlabel("iter")
    plt.savefig("acc.pdf")
    """

def final_acc():

    total=0
    correct=0
    # set to 99 to do all of them
    num_test_scenes = 10
    num_batches = 10

    # FINAL ACCURACY MEASURE
    print("FINAL ACCURACY MEASURE")
    sys.stdout.flush()
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
                    sys.stdout.flush()

                    test_data = get_events(path, scene, batch)
                    test_targets = get_targets(path, scene, batch).unsqueeze(1) 

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
                test_scene_counter += 1
        print(f"Total correctly classified test set images: {correct}/{total}")
        print(f"Test set Accuracy: {100 * correct / total:.2f}%")

#t = time.process_time()
train(counter)
iterate_through()
#elapsed = time.process_time() - t
#print(f"Training loop took {elapsed} s")

# Save network weights
#torch.save(net.state_dict(), 'snn.pt')

#load_pretrained('snn.pt')

#t = time.process_time()
#final_acc()
#elapsed = time.process_time() - t

#print(f"Final acc took {elapsed} s")

