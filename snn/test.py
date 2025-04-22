import h5py
import cv2
import numpy as np
import numba
import sys
from tqdm import tqdm
import torch
import skimage.measure
from t5 import Net

dataset_path = '/data1/fdm/eTraM/Static/HDF5/'
scene_path = 'train_h5_1/'

test_series = ['test_h5_1', 'test_h5_2']
train_series = ['train_h5_1', 'train_h5_2', 'train_h5_3', 'train_h5_4',
                'train_h5_5']

class EventList:
    def __init__(self):
        self.x = []
        self.y = []
        self.h = []
        self.w = []
        self.p = []
        self.t = []
        self.ann = []

class AnnotationFrame:
    def __init__(self):
        self.t = []
        self.x = []
        self.y = []
        self.w = []
        self.h = []
        self.class_id = []

# PROGRAM FOR PREPROCESSING OF EVENTS AND
# SAVING INTO BLOSC FILES FOR FAST TRAINING

# Hyper params
num_steps = 25
batch_size = 128
fps = 30
names = ["pedestrian", "car", "bicycle", "bus", "motorbike", "truck", "tram",
         "wheelchair"]

def set(path, scene, evlist):
    ev_path = path+scene+'_td.h5'
    ann_path = path+scene+'_bbox.npy'

    f_ev = h5py.File(ev_path, 'r')
    events = f_ev['events']
    evlist.x = events['x'][()]
    evlist.y = events['y'][()]
    evlist.h = events['height'][()]
    evlist.w = events['width'][()]
    evlist.p = events['p'][()]
    evlist.t = events['t'][()]

    evlist.ann = np.load(ann_path)

from bisect import bisect_left  
# l is list
def Binary_Search(l, x):
    i = bisect_left(l, x)
    if i:
        return (i-1)
    else:
        return -1

# Render events on image
@numba.jit(nopython=True)
def rei(image, x, y, p):
    for x_, y_, p_ in zip(x,y,p):
        if p_ == 0:
            image[y_, x_] = np.array([0,0,255])
        else:
            image[y_, x_] = np.array([255,0,0])
    return image

def draw_detection(image, bbox, label):
    """
    draw bounding box and label on the image.
    """
    bbox = list(map(int, bbox))
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image, f"{label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def render_events_and_boxes(beg, end, evlist):
    """
    beg and end of sequence in frames
    """
    ind = 0
    ann_idx = 0
    ti = beg*(1e6/fps)

    # Set previous indices
    ann_idx = Binary_Search(evlist.ann['t'], ti)
    ind_p = Binary_Search(evlist.t, ti)
    if ind_p == -1: ind_p = 0
    if ann_idx == -1: ann_idx = 0

    for i in range(beg, end):
        black_image = np.zeros((evlist.h,evlist.w,3))
        # Time in us to stop binning
        ti += 1e6/fps

        # Previous index
        ind = Binary_Search(evlist.t, ti)

        if (ind == -1): print("Ind not found")
        else: print(ind)

        # Ind is last index
        frame = rei(black_image, evlist.x[ind_p:ind+1], evlist.y[ind_p:ind+1],
                    evlist.p[ind_p:ind+1]).astype(np.uint8)
        ind_p = ind

        while (evlist.ann['t'][ann_idx] < ti):
            x1 = evlist.ann['x'][ann_idx]
            y1 = evlist.ann['y'][ann_idx]
            x2 = x1 + evlist.ann['w'][ann_idx]
            y2 = y1 + evlist.ann['h'][ann_idx]
            draw_detection(frame, (x1, y1, x2, y2), names[evlist.ann['class_id'][ann_idx]])
            ann_idx += 1
            if ann_idx >= len(evlist.ann['t']): break

        #cv2.imwrite(f"output/frame_{str(i).zfill(5)}.png", frame)
        cv2.imshow("x", frame)
        cv2.waitKey(0)

# render events on image from frame
def rei_frame(image, events):
    print(events.shape)
    print(image.shape)
    for t in tqdm(range(events.shape[0])):
        image[np.where(events[t,0] == 1)] = np.array([0,0,255])
        image[np.where(events[t,1] == 1)] = np.array([255,0,0])
        """
        for x in range(events.shape[3]):
            for y in range(events.shape[2]):
                if events[t, 0, y, x]:
                    image[y,x] = np.array([0,0,255])
                elif events[t, 1, y, x]:
                    image[y,x] = np.array([255,0,0])
        """
    return image
    """
    for x_, y_, p_ in zip(x,y,p):
        if p_ == 0:
            image[y_, x_] = np.array([0,0,255])
        else:
            image[y_, x_] = np.array([255,0,0])
    return image
    """

def render_boxes(events, anns):
    """
    Beg and End of sequence in frames
    """

    black_image = np.zeros((events.shape[2], events.shape[3], 3))

    # Ind is last index
    frame = rei_frame(black_image, events).astype(np.uint8)

    print(len(anns.t))

    for i in range(len(anns.t)):

        x1 = anns.x[i]
        y1 = anns.y[i]
        x2 = x1 + anns.w[i]
        y2 = y1 + anns.h[i]
        print(names[anns.class_id[i]])
        draw_detection(frame, (x1, y1, x2, y2), names[anns.class_id[i]])

    #cv2.imwrite(f"output/frame_{str(i).zfill(5)}.png", frame)
    cv2.imshow("x", frame)
    cv2.waitKey(0)


# Process time bin for one h5 file
# First, try saving entire file, to a file
def time_bin(beg, end, ev):
    # BEG and END are in frames
    tn = beg*1e6/fps
    ind = Binary_Search(ev.t, tn)
    if ind == -1: ind = 0
    ind_p = ind

    ann_ind = Binary_Search(ev.ann['t'], tn)
    if ann_ind == -1: ann_ind = 0
    ann_ind_p = ann_ind

    print(f" LOADING {end-beg} DATA VECTORS ".center(50, "#"))
    events = np.zeros((end-beg, num_steps, 2, ev.h, ev.w), dtype='b')
    targets = np.zeros(end-beg, dtype='b')
    anns = []

    for i in tqdm(range(beg, end)):
        ann_frame = AnnotationFrame()
        for n in range(num_steps):
            # Max time in us for bin n
            tn += 1e6/fps/num_steps
            ind_p = ind
            while(ev.t[ind] < tn):
                ind += 1
                if ind >= len(ev.t): break
            x_bin = np.clip(ev.x[ind_p:ind+1], a_min=0,
                            a_max=(events.shape[4]-1))
            y_bin = np.clip(ev.y[ind_p:ind+1], a_min=0,
                            a_max=(events.shape[3]-1))
            p_bin = np.clip(ev.p[ind_p:ind+1], a_min=0, a_max=1)
            events[i-beg, n, p_bin, y_bin, x_bin] = 1

        ann_ind = Binary_Search(ev.ann['t'], tn)
        if (ann_ind > ann_ind_p):
            # Means there is a dection in frame
            targets[i-beg] = 1
            ann_frame.t = ev.ann['t'][ann_ind_p:ann_ind+1]
            ann_frame.x = ev.ann['x'][ann_ind_p:ann_ind+1]
            ann_frame.y = ev.ann['y'][ann_ind_p:ann_ind+1]
            ann_frame.w = ev.ann['w'][ann_ind_p:ann_ind+1]
            ann_frame.h = ev.ann['h'][ann_ind_p:ann_ind+1]
            ann_frame.class_id = ev.ann['class_id'][ann_ind_p:ann_ind+1]

        ann_ind_p = ann_ind
        anns.append(ann_frame)
    
    return [events, targets, anns]

ev = EventList()
path = dataset_path+scene_path

#render_events_and_boxes(0, 500, evlist)
# Need to get the number of frames in thingy.
# Just do t[-1]*fps/1e6
# BLOSC TO THE RESCUEEEEE
# FAST AND TAKES UP LIKE NO SPACE LETS GOO
#set(path, prefix, 1, ev)
#render_events_and_boxes(2950, 3000, ev)

#beg = 100
#events, targets = time_bin(beg, beg+batch_size, ev)

import time
import blosc2
import os

#for j in range(1,21):
for series in train_series:
    path = dataset_path+series+'/'
    print(f" LOADING SERIES {series} ".center(50, "#"))
    if series == 'train_h5_1': continue
    #if series == 'train_h5_2': continue
    #if series == 'train_h5_3': continue
    #if series == 'train_h5_4': continue

    for filename in os.listdir(path):
        if filename.endswith('_td.h5'):
            scene = filename.split('_td.h5')[0]
            set(path, scene, ev)

            num_frames = int(ev.t[-1]*fps/1e6) 
            for i in range(0, num_frames-batch_size, batch_size):
                batch_num = int(i/batch_size)
                print(f"LOADING SCENE {scene}, BATCH {batch_num}")

                events, targets, anns = time_bin(i, i+batch_size, ev)
                for j in range(len(events)):
                    render_boxes(events[j], anns[j])




                # if output of model is 1, update the prediction bounding boxes
                # compare with the ground truth bounding boxes for the frame
                # using IOU



