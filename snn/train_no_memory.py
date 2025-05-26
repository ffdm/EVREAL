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

import wandb
wandb.login()

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
# Gets overwritten by wandb
num_steps = 0

spike_grad = surrogate.atan()

# Global vars for target gen
total_iou = 0
total_seen = 0
dropped = 0
min_seen_iou = 1
iou_thr = 0.5
res = []

# IOU threshold to send
min_iou = 1

# Define Network
class Net(nn.Module):
    def __init__(self, beta):
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
        global num_steps
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
    events = blosc2.load_array(path+str(num_steps)+'_steps/'+prefix+'_ev_b'+str(bn).zfill(2)+'.b2')
    return torch.Tensor(events)


def get_targets(path, prefix, bn):
    global total_iou
    global total_seen
    global dropped
    global min_seen_iou
    targets = np.zeros(batch_size)

    anns = np.load(path+str(num_steps)+'_steps/'+prefix+'_tg_b'+str(bn).zfill(2)+'.npy',
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
        else: iou_min = min_iou

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

## EVALUATION METHODS
def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou

def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (list): list of dictionary formatted like:
            [
                {'true_pos': int, 'false_pos': int, 'false_neg': int},
                {'true_pos': int, 'false_pos': int, 'false_neg': int},
                ...
            ]
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for res in img_results:
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def eval_model(net, path, prefix, bn, total, correct, dropped, ones):
    global total_iou
    global min_seen_iou
    global iou_thr
    global res

    anns = np.load(path+str(num_steps)+'_steps/'+prefix+'_tg_b'+str(bn).zfill(2)+'.npy',
                    allow_pickle=True)

    targets = get_targets(path, prefix, bn)
    events = get_events(path, prefix, bn).to(device)

    # ANNOTATION FOR PREVIOUSLY DETECTED FRAME
    ann_p = np.array([-1.0,0,0,0,0,0,0,0], dtype=anns[0].dtype)

    # Send extra one after switching
    prev = 1

    # Reset membrane potentials every batch
    mem1 = torch.zeros(1, 16, 88, 158).to(device)
    mem2 = torch.zeros(1, 32, 42, 77).to(device)
    mem3 = torch.zeros(1, 2).to(device)

    for i, ann in enumerate(anns):
        total += 1

        spk_rec,_,mem1,mem2,mem3=net(events[i].unsqueeze(0),mem1,mem2,mem3)
        _, send = spk_rec.sum(dim=0).max(1)

        if send == int(targets[i]):
            correct += 1

        if int(targets[i]) == 1:
            ones += 1

        # Second part to ensure sends extra one when switching
        if (send == 1 or (send == 0 and prev == 1)):
            ann_p = ann
            total_iou += 1

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
        
        else:

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

            if len(ious) > 0:
                iou_min = np.min(ious)
            elif ((len(bbox) == 0) and (len(bbox_p) == 0)):
                iou_min = 1
            else:
                iou_min = min_seen_iou
            
            total_iou += iou_min
            dropped += 1

            if (iou_min < min_seen_iou):
                min_seen_iou = iou_min

        prev = send

        res.append(get_single_image_results(bbox, bbox_p, iou_thr))

    return total, correct, dropped, ones

def get_targets_visual(path, prefix, bn):
    global total_iou
    global total_seen
    global dropped
    global min_seen_iou
    targets = np.zeros(batch_size)

    anns = np.load(path+str(num_steps)+'_steps/'+prefix+'_tg_b'+str(bn).zfill(2)+'.npy',
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

# Loss fn and optimizer
# loss_fn = SF.mse_membrane_loss()
#loss_fn = SF.mse_count_loss(correct_rate=1, incorrect_rate=0.1)
#loss_fn = SF.mse_count_loss()
loss_fn = SF.ce_count_loss()

# Num epochs to stop at
num_epochs = 10

# Num batches per epoch to stop at
num_batches = 20

loss_hist = []
test_acc_hist = []

counter = 0

# outer training loop
def train(iter_counter, net):
    ones = 0
    total = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-5, betas=(0.9, 0.99))

    for series in train_series:
        print(f" SERIES {series} ".center(50, "#"))
        path = data_path+series+'/'

        epoch_counter = 0
        
        scenefiles = glob(path+'*_td.h5')
        random.shuffle(scenefiles)
        for scenefile in scenefiles:
            if epoch_counter == num_epochs: break

            scene = scenefile.replace(path, '').replace('_td.h5', '')
            print(f" SCENE {scene} ".center(50, "#"))

            batch_counter = 0

            batchfiles = glob(path+str(num_steps)+'_steps/'+scene+'_tg_b*.npy')
            random.shuffle(batchfiles)
            for batchfile in batchfiles:
                if batch_counter == num_batches: break

                batch = batchfile.replace(path+str(num_steps)+'_steps/', '').replace('.npy', '')
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

                    ones += mb_ones

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
                        mem_rec[:, i] = m

                    # Membrane loss, pass last layer membranes
                    # loss_val = loss_fn(mem_rec, targets)
                    # Spike loss, pass spk_rec
                    loss_val = loss_fn(spk_rec, targets)

                    _, idx = spk_rec.sum(dim=0).max(1)

                    print("----- Inference from Minibatch ------")
                    for i in range(num):
                        print(f"Target: {targets[i].item()}, spikes: {int(spk_rec.sum(dim=0)[i][0].item())}, {int(spk_rec.sum(dim=0)[i][1].item())}")

                    # grad calc + weight update
                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()
            
                    # minibatch accuracy (before weight update)
                    acc = np.mean((targets == idx).detach().cpu().numpy())
                    print(f"Percent of ones in minibatch: {100*mb_ones/num:.2f}%")
                    print(f"Minibatch accuracy: {100*acc:.2f}%")
                    print(f"Minibatch loss: {loss_val.item():.2f}")

                    wandb.log(
                        {
                            "minibatch acc": acc,
                            "minibatch loss": loss_val.item(),
                        }
                    )

                    iter_counter += 1

                batch_counter += 1

            epoch_counter += 1

    print(f"Total ones in training data: {ones}/{total}")

def final_acc(net):

    total=0
    correct=0
    dropped=0
    ones=0

    # set to 99 to do all of them
    num_test_scenes = 20
    num_test_batches = 20

    # FINAL ACCURACY MEASURE
    # Target accuracy
    # Percent dropped
    # Precision
    
    print("FINAL ACCURACY MEASURE")
    with torch.no_grad():
        net.eval()
        for series in test_series:
            print(f" RUNNING SERIES {series} ".center(50, "#"))
            path = data_path+series+'/'

            test_scene_counter = 0

            scenefiles = glob(path+'*_td.h5')
            random.shuffle(scenefiles)
            for scenefile in scenefiles:
                scene = scenefile.replace(path, '').replace('_td.h5', '')

                if test_scene_counter == num_test_scenes: break

                print(f" RUNNING SCENE {scene} ".center(50, "#"))

                test_batch_counter = 0

                batchfiles = glob(path+str(num_steps)+'_steps/'+scene+'_tg_b*.npy')
                random.shuffle(batchfiles)
                for batchfile in batchfiles:
                    if test_batch_counter == num_test_batches: break

                    batch = batchfile.replace(path+str(num_steps)+'_steps/', '').replace('.npy', '')
                    batch = int(batch.replace(scene+'_tg_b', ''))
                    print(f" Batch {test_batch_counter} ({batch}) ".center(50, "-"))
                        
                    total, correct, dropped, ones = eval_model(net, path, scene,
                                                               batch, total,
                                                               correct,
                                                               dropped, ones)
                    test_batch_counter += 1
                test_scene_counter += 1
        precision, recall = calc_precision_recall(res)
        wandb.log({
            "acc": (100*correct/total), 
            "precision": precision,
            "recall": recall,
            "dropped": dropped/total, 
            "ones": ones/total, 
            "score": precision*(dropped/total)
        })

        print(f"Total correctly classified test set images: {correct}/{total}")
        print(f"Test set Accuracy: {100 * correct / total:.2f}%")
        print(f"Total ones in test set: {ones}")
        print(f"Guessing Accuracy : {100 * (1 - ones / total):.2f}%")

run_cnt = 0
def run():
    global run_cnt
    global num_steps
    wandb.init()

    beta = wandb.config['beta']
    wandb.log({"beta":beta})

    num_steps = 4
    #num_steps = wandb.config['num_steps']
    #wandb.log({"num_steps":num_steps})

    net = Net(beta).to(device)

    start = time.time()
    train(counter, net)
    #iterate_through()
    end = time.time()
    print(f"Training loop took {end-start:.2f} s")

    # Save network weights
    torch.save(net.state_dict(), 'snn_no_rst'+str(run_cnt)+'.pt')

    start = time.time()
    final_acc(net) # target accuracy
    end = time.time()
    print(f"Final acc took {end-start:.2f} s")

    run_cnt += 1


config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "acc"},
    "parameters": {
        "beta": {
            "min": 0.9,
            "max": 1.0
        },
        #"num_steps": {"values": [1, 2, 4, 8]},
    },
}

sweep_id = wandb.sweep(sweep=config, project="4step_1iou")

wandb.agent(sweep_id, function=run, count=3)
