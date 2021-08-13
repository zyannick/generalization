import multiprocessing as mp

import os
import numpy as np
import skvideo.io
import densetrack
from glob import glob
from termcolor import colored
from datetime import datetime
from time import time



root_dir = 'five_fps_cme_sep'
save_dir = 'five_fps_cme_sep_idt'


def generate_files(video_file_name):
    head, tail = os.path.split(video_file_name)
    name = os.path.splitext(tail)[0]
    print(colored( '{}  --> Starting {} at {}'.format( mp.current_process(), name, datetime.fromtimestamp(int(time()))), 'green'))
    debut = time()
    video_frames = skvideo.io.vreader(fname=video_file_name, as_grey=True)
    video_gray = np.stack([np.reshape(x, x.shape[1:3])
                        for x in video_frames]).astype(np.uint8, copy=False)
    tracks = densetrack.densetrack(video_gray, adjust_camera=True)
    
    np.save(os.path.join(save_dir, name + '-traj'), tracks)

    fin = time()
    print(colored( '{} --> Ending {} at {} for a duration of  {} seconds'.format(mp.current_process(), name, datetime.fromtimestamp(int(time())), int(fin - debut)), 'blue'))



if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    list_files = glob(os.path.join(root_dir, '*.avi'))

    print(len(list_files))
    print(mp.cpu_count())

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(generate_files, list_files)
    pool.close()
    pool.join()