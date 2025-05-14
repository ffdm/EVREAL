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
min_iou = 0.8

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
    
    def forward(self, x, mem1, mem2, mem3):

        # Record final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            # Do the maxpool before storing the data to decrease datasize
            cur1 = F.max_pool2d(x[:, step], 4)
            cur1 = F.max_pool2d(self.conv1(cur1),2)
            #spk1, self.mem1 = self.lif1(cur1, self.mem1)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = F.max_pool2d(self.conv2(spk1),2)
            #spk2, self.mem2 = self.lif2(cur2, self.mem2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = torch.flatten(spk2, 1)

            cur3 = self.fc1(spk2)
            #spk3, self.mem3 = self.lif3(cur3, self.mem3)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0), \
                mem1, mem2, mem3

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

def draw_bbox_array(image, bbox, label):
    """
    Draw bounding box and label on the image.
    """
    print(bbox.shape) 
    print(label)
    print(bbox)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        if type(label) is not str:
            # MEANS NOT SENT
            cv2.putText(image, f"IoU: {label[i]:.2f}", (x2 - 80, y2 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif label == 'send':
            cv2.putText(image, f"IoU: 1.00", (x2 - 75, y2 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_send(image, send):
    """
    image: Image to draw
    send: Boolean for if to send or not
    """
    h = image.shape[0]
    w = image.shape[1]
    if send:
        cv2.rectangle(image, (0,0), (w, h), (0,255,0), 5)
        cv2.putText(image, "Sent", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.rectangle(image, (0,0), (w, h), (0,0,255), 5)
        cv2.putText(image, "Skipped", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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

def render_frame_on_image(events, targets, image):

    print(events.shape)
    print(image.shape)

    frame = rei_frame(image, events)

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


def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0):
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


def get_targets(path, prefix, bn):
    global total_iou
    global total_seen
    global dropped
    global min_seen_iou
    targets = np.zeros(batch_size)

    anns = np.load(path+'b2/'+prefix+'_tg_b'+str(bn).zfill(2)+'.npy',
                    allow_pickle=True)

    # ANNOTATION FOR PREVIOUSLY DETECTED FRAME
    ann_p = np.array([-1.0,0,0,0,0,0,0,0], dtype=anns[0].dtype)

    for i, ann in enumerate(anns):
       
        total_seen += 1

        ## ONLY DETECT OBJECT OR NOT FOR DEBUGGING
        """
        if len(ann) > 0:
            targets[i] = 1

        continue
        """

        ## ... might need a recurrent model otherwise ...
        ## and what should I do on the first input of the batch?

        if len(ann_p) != len(ann) or i == 0:

            # SNN Sends if # anns change or first frame
            targets[i] = 1
            ann_p = ann
            total_iou += 1

            continue

        x1_p = ann_p['x']
        y1_p = ann_p['y']
        x2_p = x1_p + ann_p['w']
        y2_p = y1_p + ann_p['h']
        bbox_p = np.column_stack((x1_p, y1_p, x2_p, y2_p))

        x1 = ann['x']
        y1 = ann['y']
        x2 = x1 + ann['w']
        y2 = y1 + ann['h']
        bbox = np.column_stack((x1, y1, x2, y2))

        (idxs_true, idxs_pred, ious, labels) = match_bboxes(bbox, bbox_p) 

        if len(ious) > 0: iou_min = np.min(ious)
        else: 
            iou_min = min_seen_iou

        if iou_min < min_iou or len(ious) < len(ann):
            targets[i] = 1
            ann_p = ann
            total_iou += 1

        else:
            total_iou += iou_min
            dropped += 1

            if (min_seen_iou > iou_min):
                min_seen_iou = iou_min

    return torch.Tensor(targets)


def get_targets_visual(path, prefix, bn):
    global total_iou
    global total_seen
    global dropped
    global min_seen_iou
    targets = np.zeros(batch_size)

    anns = np.load(path+'b2/'+prefix+'_tg_b'+str(bn).zfill(2)+'.npy',
                    allow_pickle=True)
    events = get_events(path, prefix, bn)

    # TARGET GENERATION METHODOLOGY
    # FIRST IMAGE ALWAYS IS MARKED AS A 1 to encourage SNN to send more
    # often than not. The target is then saved. 
    # If the number of BBs changes? SNN sends
    # Else, the iou is calciulated from previous send
    # If it is below a threshold? the SNN sends

    # ANNOTATION FOR PREVIOUSLY DETECTED FRAME
    ann_p = np.array([-1.0,0,0,0,0,0,0,0], dtype=anns[0].dtype)

    for i, ann in enumerate(anns):
        total_seen += 1

        # black image
        image = np.zeros((events.shape[3], events.shape[4], 3)) 

        x1_p = ann_p['x']
        y1_p = ann_p['y']
        x2_p = x1_p + ann_p['w']
        y2_p = y1_p + ann_p['h']
        bbox_p = np.column_stack((x1_p, y1_p, x2_p, y2_p))

        x1 = ann['x']
        y1 = ann['y']
        x2 = x1 + ann['w']
        y2 = y1 + ann['h']
        bbox = np.column_stack((x1, y1, x2, y2))

        if len(ann_p) != len(ann) or i == 0:

            # SNN Sends if # anns change or first frame
            targets[i] = 1
            ann_p = ann
            total_iou += 1

            draw_bbox_array(image, bbox, 'send')
            draw_send(image, 1)

            cv2.putText(image, 
            f"{dropped/total_seen:.3f}% dropped",
            (1000, 710), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            render_frame_on_image(events[i], ann, image) 

            continue

        (idxs_true, idxs_pred, ious, labels) = match_bboxes(bbox, bbox_p) 

        if len(ious) > 0: iou_min = np.min(ious)
        else: 
            iou_min = min_seen_iou

        if iou_min < min_iou or len(ious) < len(ann):
            targets[i] = 1
            ann_p = ann
            total_iou += 1

            draw_bbox_array(image, bbox, 'send')
            draw_send(image, 1)

            cv2.putText(image, 
            f"{dropped/total_seen:.3f}% dropped",
            (1000, 710), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        else:
            total_iou += iou_min
            dropped += 1

            draw_bbox_array(image, bbox_p, ious)
            draw_send(image, 0)

            cv2.putText(image, 
            f"{dropped/total_seen:.3f}% dropped",
            (1000, 710), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if (min_seen_iou > iou_min):
                min_seen_iou = iou_min

        render_frame_on_image(events[i], ann, image) 

    return torch.Tensor(targets)

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

                #test_data = get_events(path, scene, batch)
                test_targets = get_targets(path, scene, batch).unsqueeze(1) 

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
num_batches = 5

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

                    spk_rec = torch.zeros(num_steps,num,2).to(device)
                    mem_rec = torch.zeros(num_steps,num,2).to(device)

                    mem1 = torch.zeros(1, 16, 88, 158).to(device)
                    mem2 = torch.zeros(1, 32, 42, 77).to(device)
                    mem3 = torch.zeros(1, 2).to(device)

                    for i, d in enumerate(data):
                        sp, m, mem1, mem2, mem3 = net(d.unsqueeze(0), mem1, mem2, mem3)
                        sp = sp.squeeze(1)
                        m = m.squeeze(1)
                        spk_rec[:, i] = sp

                    # loss
                    loss_val = loss_fn(spk_rec, targets)
                    _, idx = spk_rec.sum(dim=0).max(1)

                    print("----- 5 Points from Minibatch ------")
                    for i in range(5):
                        print(f"Target: {targets[i].item()}, spikes: {int(spk_rec.sum(dim=0)[i][0].item())}, {int(spk_rec.sum(dim=0)[i][1].item())}")

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
                    #loss_val.backward(retain_graph=True)
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
    plt.savefig("loss_no_rst.pdf")
    plt.clf()
    plt.plot(test_acc_hist)
    plt.xlabel("iter")
    plt.ylabel("minibatch acc")
    plt.savefig("acc_no_rst.pdf")

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

                    net.reset()

                    minibatches = 128
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
torch.save(net.state_dict(), 'snn_no_rst.pt')
start = time.time()
final_acc()
end = time.time()
print(f"Final acc took {end-start:.2f} s")

#load_pretrained('snn.pt')
