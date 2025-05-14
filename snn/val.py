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
correct = 0
min_seen_iou = 1
res = []

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

def draw_bbox_array(image, bbox, label):
    """
    Draw bounding box and label on the image.
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        if type(label) is not str:
            # MEANS NOT SENT
            if i < len(label):
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

def render_frame_on_image(events, targets, image):

    frame = rei_frame(image, events)

    for ann in targets:
        x1 = ann['x']
        y1 = ann['y']
        x2 = x1 + ann['w']
        y2 = y1 + ann['h']
        label = f"{ann['class_confidence']:.2f}: {objects[ann['class_id']]}"
        draw_detection(frame, (x1, y1, x2, y2), label)
    
    return frame

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
    return torch.Tensor(events).to(device)

def get_targets(path, prefix, bn):
    global total_iou
    global dropped
    global min_seen_iou
    targets = np.zeros(batch_size)
    
    anns = np.load(path+'b2/'+prefix+'_tg_b'+str(bn).zfill(2)+'.npy',
                    allow_pickle=True)

    # ANNOTATION FOR PREVIOUSLY DETECTED FRAME
    ann_p = np.array([-1.0,0,0,0,0,0,0,0], dtype=anns[0].dtype)

    for i, ann in enumerate(anns):

        if len(ann) > 0:
            targets[i] = 1

        continue

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



def eval_visual(path, prefix, bn):
    global total_iou
    global total_seen
    global dropped
    global min_seen_iou
    global correct 

    anns = np.load(path+'b2/'+prefix+'_tg_b'+str(bn).zfill(2)+'.npy',
                    allow_pickle=True)
    events = get_events(path, prefix, bn)
    targets = get_targets(path, prefix, bn)

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

        spk_rec, _ = net(events[i].unsqueeze(0))
        _, send = spk_rec.sum(dim=0).max(1)

        if send == int(targets[i]):
            correct += 1

        cv2.putText(image, 
        f"Target: {int(targets[i])}",
        (600, 710), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(image, 
        f"Correct: {correct}/{total_seen} = {100*correct/total_seen:.2f}%",
        (10, 710), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if (send == 1):
            ann_p = ann
            total_iou += 1
            draw_bbox_array(image, bbox, 'send')
            draw_send(image, 1)

        else:

            (idxs_true, idxs_pred, ious, labels) = match_bboxes(bbox, bbox_p) 

            if len(ious) > 0: 
                iou_min = np.min(ious)
                draw_bbox_array(image, bbox_p, ious)
            else: 
                iou_min = min_seen_iou

            total_iou += iou_min
            dropped += 1

            draw_send(image, 0)

            if (min_seen_iou > iou_min):
                print(f"New Min IoU: {iou_min}")
                min_seen_iou = iou_min

        cv2.putText(image, 
        f"{100*dropped/total_seen:.2f}% dropped",
        (1000, 710), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = render_frame_on_image(events[i].cpu(), ann, image) 
        evalname = f'./evalframes/{str(total_seen).zfill(5)}_{prefix}_b{bn}_{str(i).zfill(3)}'
        #cv2.imwrite(evalname+'.png', frame)
        #cv2.imshow("Frame", frame)
        #cv2.waitKey(0)


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

            frame = render_frame_on_image(events[i].cpu(), ann, image) 
            outname = f'./targetframes/{str(total_seen).zfill(5)}_{prefix}_b{bn}_{str(i).zfill(3)}'
            cv2.imwrite(outname+'.png', frame)

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

        frame = render_frame_on_image(events[i].cpu(), ann, image) 
        outname = f'./targetframes/{str(total_seen).zfill(5)}_{prefix}_b{bn}_{str(i).zfill(3)}'
        cv2.imwrite(outname+'.png', frame)

    return torch.Tensor(targets)


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

def plot_pr_curve(
    precisions, recalls, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax

def eval_model(path, prefix, bn):
    global total_iou
    global total_seen
    global dropped
    global min_seen_iou
    global res

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

    # Send an extra one after switching
    prev = 1

    for i, ann in enumerate(anns):
        total_seen += 1

        spk_rec, _ = net(events[i].unsqueeze(0))
        _, send = spk_rec.sum(dim=0).max(1)

        #if (send == 1):
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

        res.append(get_single_image_results(bbox, bbox_p, 0.95))


def iterate_through():
    for series in test_series:
        path = data_path+series+'/'

        test_scene_counter = 0

        print(f" Iterating series {series} ".center(50, "#"))

        scenefiles = glob(path+'*_td.h5')
        random.shuffle(scenefiles)
        for scenefile in scenefiles:
            scene = scenefile.replace(path, '').replace('_td.h5', '')

            #if test_scene_counter == 10: break

            print(f" Iterating scene {scene} ".center(50, "#"))

            batch_counter = 0

            batchfiles = glob(path+'b2/'+scene+'_tg_b*.npy')
            random.shuffle(batchfiles)
            for batchfile in batchfiles:
                batch = batchfile.replace(path+'b2/', '').replace('.npy', '')
                batch = int(batch.replace(scene+'_tg_b', ''))

                #if batch_counter == 5: break

                print(f" Iterating batch {batch_counter} ({batch}) ".center(50, "#"))

                #test_data = get_events(path, scene, batch)
                #test_targets = get_targets(path, scene, batch).unsqueeze(1) 
                #eval_visual(path, scene, batch)
                eval_model(path, scene, batch)
                #get_targets_visual(path, scene, batch)

                batch_counter += 1

            test_scene_counter += 1

        print(calc_precision_recall(res))
        print(f"Avg IOU: {total_iou/total_seen}, lowest IOU: {min_seen_iou}")
        print(f"dropped: {dropped}/{total_seen}={dropped/total_seen:.4f}")

load_pretrained('snn_no_rst_old.pt')
start = time.time()
iterate_through()
end = time.time()
print(f"Eval took {end-start:.2f} s")

