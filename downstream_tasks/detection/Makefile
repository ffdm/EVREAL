ecd = boxes_6dof calibration dynamic_6dof office_zigzag poster_6dof shapes_6dof slider_depth
etram = train_h5_1

all: $(etram)

$(etram):
	python detect.py --input ../../outputs/t50ms/eTraM_npy/$@/HyperE2VID/ --output outputs/HyperE2VID/$@ 
	#cat outputs/HyperE2VID/$@/drawings/* | ffmpeg -framerate 10 -f image2pipe -y -i - ~/out/$@_detect.mp4

$(ecd):
	python detect.py --input ../../outputs/std/ECD/$@/HyperE2VID/ --output outputs/HyperE2VID/$@ 
	cat outputs/HyperE2VID/$@/drawings/* | ffmpeg -f image2pipe -y -i - ~/out/$@.mp4

