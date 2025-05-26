import h5py
import cv2
import numpy as np
import numba
import sys
from tqdm import tqdm
import torch
import skimage.measure

sys.settrace

dataset_path = '/data1/fdm/eTraM/Static/HDF5/'

test_series = ['test_h5_1', 'test_h5_2']
train_series = ['train_h5_1', 'train_h5_2', 'train_h5_3', 'train_h5_4',
                'train_h5_5']

class EventList:
    def init(self):
        self.x = []
        self.y = []
        self.h = []
        self.w = []
        self.p = []
        self.t = []
        self.ann = []

class AnnList:
    def init(self):
        self.x = []
        self.y = []
        self.w = []
        self.h = []
        self.c = [] # class_id

# PROGRAM FOR PREPROCESSING OF EVENTS AND
# SAVING INTO BLOSC FILES FOR FAST TRAINING

# Hyper params
num_steps = 1
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
        label = f"{ann['class_confidence']:.2f}: {names[ann['class_id']]}"
        draw_detection(frame, (x1, y1, x2, y2), label)

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)


def render_events_and_boxes(beg, end, evlist):
    """
    Beg and End of sequence in frames
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

def target_bin(beg, end, ev):
    tn = beg*1e6/fps
    ann_ind = Binary_Search(ev.ann['t'], tn)
    if ann_ind == -1: ann_ind = 0
    ann_ind_p = ann_ind

    print(f" LOADING {end-beg} TARGETS ".center(50, "#"))
    targets = np.empty(end-beg, dtype=object) 

    for i in tqdm(range(beg, end)):
        targets[i-beg] = np.array([], dtype=ev.ann.dtype)

        for n in range(num_steps):
            # Max time in us for bin n
            tn += 1e6/fps/num_steps

        ann_ind = Binary_Search(ev.ann['t'], tn)
        if ann_ind == -1: ann_ind = 0
        if (ann_ind > ann_ind_p):
            targets[i-beg] = ev.ann[ann_ind_p:ann_ind]
        ann_ind_p = ann_ind
    
    return targets

# Process time bin for one h5 file
# First, try saving entire file, to a file
def time_bin(beg, end, ev):
    # BEG and END are in frames
    tn = beg*1e6/fps
    ind = Binary_Search(ev.t, beg*1e6/fps)
    if ind == -1: ind = 0
    ind_p = ind

    ann_ind = Binary_Search(ev.ann['t'], tn)
    if ann_ind == -1: ann_ind = 0
    ann_ind_p = ann_ind

    print(f" LOADING {end-beg} DATA VECTORS ".center(50, "#"))
    events = np.zeros((end-beg, num_steps, 2, ev.h, ev.w), dtype='b')
    targets = np.empty(end-beg, dtype=object) 

    for i in tqdm(range(beg, end)):
        targets[i-beg] = np.array([], dtype=ev.ann.dtype)

        for n in range(num_steps):
            # Max time in us for bin n
            tn += 1e6/fps/num_steps
            ind_p = ind
            while(ev.t[ind] < tn):
                ind += 1
                if ind >= len(ev.t): break
            x_bin = np.clip(ev.x[ind_p:ind], a_min=0,
                            a_max=(events.shape[4]-1))
            y_bin = np.clip(ev.y[ind_p:ind], a_min=0,
                            a_max=(events.shape[3]-1))
            p_bin = np.clip(ev.p[ind_p:ind], a_min=0, a_max=1)
            events[i-beg, n, p_bin, y_bin, x_bin] = 1

        ann_ind = Binary_Search(ev.ann['t'], tn)
        if ann_ind == -1: ann_ind = 0
        if (ann_ind > ann_ind_p):
            targets[i-beg] = ev.ann[ann_ind_p:ann_ind]
            """
            ann = (np.array([ev.ann['x'][ann_ind_p:ann_ind+1],
                          ev.ann['y'][ann_ind_p:ann_ind+1],
                          ev.ann['w'][ann_ind_p:ann_ind+1],
                          ev.ann['h'][ann_ind_p:ann_ind+1],
                          ev.ann['class_id'][ann_ind_p:ann_ind+1],
                          ev.ann['track_id'][ann_ind_p:ann_ind+1],
                          ev.ann['class_confidence'][ann_ind_p:ann_ind+1]],
                                    dtype=np.float32))
        else:
            ann = np.array([-1,-1,-1,-1,-1,-1,-1],dtype=np.float32) 
        targets.append(ann)
            """
        ann_ind_p = ann_ind
    
    #return [events, np.array(targets)]
    return [events, targets]

ev = EventList()

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
for series in train_series+test_series:
    path = dataset_path+series+'/'
    outpath = path+str(num_steps)+'_steps'+'/'
    print(f" LOADING SERIES {series} ".center(50, "#"))
    #if series == 'train_h5_5': continue
    #if series == 'test_h5_1': continue

    for filename in os.listdir(path):
        if filename.endswith('_td.h5'):
            scene = filename.split('_td.h5')[0]
            set(path, scene, ev)
            os.makedirs(outpath+'b2/', exist_ok=True)
            num_frames = int(ev.t[-1]*fps/1e6) 
            for i in range(0, num_frames-batch_size, batch_size):
                batch_num = int(i/batch_size)
                print(f"LOADING SCENE {scene}, BATCH {batch_num}")
                events, targets = time_bin(i, i+batch_size, ev)
                targets = target_bin(i, i+batch_size, ev)
                #render(events, targets)

                

                events_outfile = outpath+'b2/'+scene+'_ev_b'+str(batch_num).zfill(2)+'.b2'
                targets_outfile = outpath+'b2/'+scene+'_tg_b'+str(batch_num).zfill(2)+'.npy'

                blosc2.save_tensor(events, events_outfile, mode='w')
                np.save(targets_outfile, targets)
            
sys.exit()

start = time.perf_counter()
events = blosc2.load_array(events_outfile)
targets = blosc2.load_array(targets_outfile)
end = time.perf_counter()
print(f"Blosc load time: {end-start}")
print(events.shape)
print(targets.shape)

sys.exit()

#np.save(f, events)
#f.close()
#np.savez(outfile, events, targets)

sys.exit()
#np.save(outfile, 

def save_for_EVREAL():
    events_ts_path = '../../../../eTraM_npy/train_h5_1/events_ts.npy' 
    events_xy_path = '../../../../eTraM_npy/train_h5_1/events_xy.npy' 
    events_p_path = '../../../../eTraM_npy/train_h5_1/events_p.npy' 
    t_new = t / float(1e6)

    xy = np.transpose(np.array([x,y]))
    print(xy.shape)

    np.save(events_ts_path, t, allow_pickle=False, fix_imports=False)
    np.save(events_xy_path, xy, allow_pickle=False, fix_imports=False)
    np.save(events_p_path, p, allow_pickle=False, fix_imports=False)




