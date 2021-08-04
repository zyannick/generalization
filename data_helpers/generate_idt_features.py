import multiprocessing as mp

import os
import numpy as np
import skvideo.io
import densetrack


video_frames = skvideo.io.vreader(fname=video_file_name, as_grey=True)
video_gray = np.stack([np.reshape(x, x.shape[1:3])
                       for x in video_frames]).astype(np.uint8, copy=False)
tracks = densetrack.densetrack(video_gray, adjust_camera=True)
head, tail = os.path.split(video_file_name)
name = os.path.splitext(tail)[0]
np.save(os.path.join(data_dir, name + '-traj'), tracks)