from loading.utils_edges import *
import cv2 as cv2
from torch.autograd import Variable

import pickle as pkl

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path
import glob
import math
from random import gauss
import torch



def getratio(frame_width = 80, frame_height = 60):
    radius_frame = math.sqrt(frame_height * frame_height + frame_width * frame_width)
    radius_80_60 = math.sqrt(60 * 60 + 80 * 80)
    return  radius_frame / radius_80_60

def load_frames(root, video_file, start, nb_frames_per_shot,
                data_aug, affine_transform, is_train, domain_key,
                modality='visible', verbose = False, original_mode = 'rgb', middle_mode = 'rgb',
                frame_width = 80, frame_height = 60):
    ratio = getratio(frame_width, frame_height)
    original_mode = original_mode
    r_f = random.randint(1, 10)
    frames = []
    file_name = root + '/' + video_file
    if verbose:
        print(file_name)
    vid_visible = cv2.VideoCapture(file_name)
    # nf = total_number_frames
    cp = 0
    insert_rectangle = random.randint(1, 2)
    rw = random.randint(5, frame_width)
    rh = random.randint(5, frame_height)

    ecart_rectangle = random.randint(int(5*ratio), int(10*ratio))

    red = 0
    green = 0
    blue = 0

    if modality == 'visible':
        red = random.randint(50, 255)
        green = random.randint(50, 255)
        blue = random.randint(50, 255)
    elif modality == 'tir' or modality == 'edge':
        pix_value = random.randint(50, 255)
        red = pix_value
        green = pix_value
        blue = pix_value

    cp = 0

    if not is_train:
        data_aug = False
        affine_transform = False

    while cp < start and vid_visible.isOpened():
        _, img = vid_visible.read()
        cp = cp + 1

    ts = 0

    delta = np.zeros(6)
    for i in range(delta.shape[0]):
        delta[i] = gauss(0, 20)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([
        [50 + delta[0], 50 + delta[1]],
        [200 + delta[2], 50 + delta[3]],
        [50 + delta[4], 200 + delta[5]]])

    if verbose:
        print("here -1 {:d}   {:d}  {}".format(ts, nb_frames_per_shot, str(vid_visible.isOpened())))
    while ts < nb_frames_per_shot and vid_visible.isOpened():
        if verbose:
            print("here 0")
        ret, img = vid_visible.read()

        if not ret:
            print('problem ret  {}'.format(file_name))
            break
        ts = ts + 1
        img = cv2.resize(img, dsize=(frame_width, frame_height))
        #if verbose:
        #    print("here 1")
        h, w, _ = img.shape
        if data_aug:
            if r_f > 5:
                img = cv2.flip(img, 1)
            dx = int(random.uniform(-1, 1) * ratio)
            dy = int(random.uniform(-1, 1) * ratio)
            if insert_rectangle == 2:
                if modality != 'edge':
                    cv2.rectangle(img, (rh + dx, rw + dy), (rh + dx + ecart_rectangle, rw + dy + ecart_rectangle),
                                  (red, green, blue), -1)
                else:
                    cv2.rectangle(img, (rh + dx, rw + dy), (rh + dx + ecart_rectangle, rw + dy + ecart_rectangle),
                                  (red, green, blue), 1)
            if affine_transform:
                M = cv2.getAffineTransform(pts1, pts2)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if domain_key == 'rgb':
            img = img
        elif domain_key == 'sobel':
            img = transform_img_sobel(img, 0, 3)
        elif domain_key == 'laplace':
            img = laplace_dev(img, 0, 3)

        #if verbose:
        #    print("here 3")
        img = img[:, :, [2, 1, 0]]
        if middle_mode == 'raw':
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            if original_mode == 'rgb':
                img = (img / 255.) * 2 - 1
        elif middle_mode == 'flow':
            img = cv2.resize(img, dsize=(256, 192))
        frames.append(img)
    #frames = extract_optical_flow_from_list(frames)
    if verbose:
        print('we will see here %d' %(len(frames)))
    return np.asarray(frames, dtype=np.float32), frames


def extract_optical_flow_from_list(list_img):
    list_flow = []

    for i in np.arange(0, len(list_img) - 1, 1):
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        prvs = list_img[i]
        next = list_img[i + 1]

        prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

        flow = optical_flow.calc(prvs, next, None)
        flow = normalize_flow(flow)
        flow = np.nan_to_num(flow)
        flow = flow * 2 - 1
        list_flow.append(flow)

    return list_flow

def extract_optical_flow_from_list_flownet(list_img, flownet):
    list_flow = []

    flownet.cuda()

    for i in np.arange(0, len(list_img) - 1, 1):
        prvs = list_img[i]
        h, w, c = prvs.shape
        prvs = prvs.reshape((1, c, h, w))
        prvs = torch.Tensor(prvs)
        prvs = Variable(prvs.cuda())

        next = list_img[i + 1]
        next = next.reshape((1, c, h, w))
        next = torch.Tensor(next)
        next = Variable(next.cuda())


        flow_ref, conf_ref = flownet(prvs, next)

        flow_ref = flow_ref.cuda()
        flow_ref = flow_ref.cpu().detach().numpy()

        flow_ref = normalize_flow(flow_ref)
        flow_ref = np.nan_to_num(flow_ref)

        print(flow_ref.shape)

        list_flow.append(flow_ref)

        #prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
        #next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)


    return list_flow


def get_rect(rw, rh, dx, dy):
    x_rect = np.zeros((rh, rw))
    y_rect = np.zeros((rh, rw))
    for i in range(rh):
        for j in range(rw):
            x_rect[i, j] = dx + random.uniform(-0.02, 0.02)
            y_rect[i, j] = dy + random.uniform(-0.02, 0.02)
    return x_rect, y_rect


def normalize_flow(flow_frame):
    h, w, _ = flow_frame.shape

    if np.max(flow_frame[:, :, 0]) - np.min(flow_frame[:, :, 0]) != 0:
        flow_frame[:, :, 0] = (flow_frame[:, :, 0] - np.min(flow_frame[:, :, 0])) / (
                    np.max(flow_frame[:, :, 0]) - np.min(flow_frame[:, :, 0]))
    else:
        flow_frame[:, :, 0] = np.zeros((h, w))

    if np.max(flow_frame[:, :, 1]) - np.min(flow_frame[:, :, 1]) != 0:
        flow_frame[:, :, 1] = (flow_frame[:, :, 1] - np.min(flow_frame[:, :, 1])) / (
                    np.max(flow_frame[:, :, 1]) - np.min(flow_frame[:, :, 1]))
    else:
        flow_frame[:, :, 1] = np.zeros((h, w))

    return flow_frame


def load_flow_frames(root_dir, vid, start, nb_frames, frame_width = 80, frame_height = 60):
    # get the shape
    dir_name = vid.split('.')[0]
    list_pkl = glob.glob(os.path.join(root_dir, dir_name, '*.pkl'))
    first_pkl = list_pkl[0]
    with open(first_pkl, "rb") as fout:
        first_flow = pkl.load(fout)
    h, w, _ = first_flow.shape

    frames = []
    rw = random.randint(5, int(math.sqrt(h * h + w * w) / 3))
    rh = random.randint(5, int(math.sqrt(h * h + w * w) / 3))
    insert_rectangle = random.randint(1, 2)

    start_x = random.randint(5, int(math.sqrt(h * h + w * w) / 3))
    start_y = random.randint(5, int(math.sqrt(h * h + w * w) / 3))

    dx_prev = 0
    dy_prev = 0
    iteration = 0

    for i in range(start, start + nb_frames):
        file_name = os.path.join(root_dir, dir_name, str(i).zfill(8) + '.pkl')
        with open(file_name, "rb") as fout:
            flow_frame = pkl.load(fout)

        if iteration == 0:
            dx_prev = random.uniform(np.min(flow_frame[:, :, 0]), np.max(flow_frame[:, :, 0])) / 4
            dy_prev = random.uniform(np.min(flow_frame[:, :, 1]), np.max(flow_frame[:, :, 1])) / 4
            dx = dx_prev
            dy = dy_prev
        else:
            dx = dx_prev + random.uniform(-1, 1)
            dy = dy_prev + random.uniform(-1, 1)
            dx_prev = dx
            dy_prev = dy

        x_rect, y_rect = get_rect(w, h, dx, dy)

        if iteration == 0:
            start_x = max(0, start_x + random.randint(-1, 1))
            start_y = max(0, start_y + random.randint(-1, 1))
        else:
            start_x = start_x + dx_prev
            start_y = start_y + dy_prev

        # print('values dx {:f}     dy  {:f}'.format(dx, dy))

        if insert_rectangle == 1:
            # print('sx {:f}   ex {:f}'.format(start_x, min(start_x + rw, w)))
            # print('sy : {:f}  ey {:f} \n\n'.format(start_y, min(start_y + rh, h)))
            sx = int(start_x)
            ex = int(min(start_x + rw, w))
            sy = int(start_y)
            ey = int(min(start_y + rh, h))
            flow_frame[sy: ey, sx: ex, 0] = x_rect[sy: ey, sx: ex]
            flow_frame[sy: ey, sx: ex, 1] = y_rect[sy: ey, sx: ex]

        # np.seterr(divide='ignore', invalid='ignore') dividing by zero
        flow_frame = normalize_flow(flow_frame)
        flow_frame = np.nan_to_num(flow_frame)

        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            flow_frame = cv2.resize(flow_frame, dsize=(0, 0), fx=sc, fy=sc)

        flow_frame = (flow_frame) * 2 - 1
        frames.append(flow_frame)
        iteration = iteration + 1
    # print(len(frames))
    return np.asarray(frames, dtype=np.float32)


def load_computed_flow(root, video_file, start, nb_frames_per_shot, edge_type,
                       total_number_frames, blur_kernel, operator_kernel,
                       data_aug, affine_transform, is_inference, frame_width = 80, frame_height = 60):
    r_f = random.randint(1, 10)

    frames = []
    file_name = root + '/' + video_file

    # print(video_file)

    vid_visible = cv2.VideoCapture(file_name)
    # nf = total_number_frames

    cp = 0

    c_operator = random.randint(1, 2)
    # c_blur = random.randint(1,2)

    insert_rectangle = random.randint(1, 2)
    rw = random.randint(5, frame_width)
    rh = random.randint(5, frame_height)

    ecart_rectangle = random.randint(5, 10)

    red = random.randint(50, 255)
    green = random.randint(50, 255)
    blue = random.randint(50, 255)

    if edge_type == 'normal':
        # c_blur = 0
        c_operator = 0

    cp = 0

    if is_inference:
        data_aug = False
        affine_transform = False

    while cp < start and vid_visible.isOpened():
        _, img = vid_visible.read()
        cp = cp + 1

    ts = 0

    delta = np.zeros(6)
    for i in range(delta.shape[0]):
        delta[i] = gauss(0, 3.5)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([
        [50 + delta[0], 50 + delta[1]],
        [200 + delta[2], 50 + delta[3]],
        [50 + delta[4], 200 + delta[5]]])

    while ts < nb_frames_per_shot and vid_visible.isOpened():
        ret, img = vid_visible.read()

        if not ret:
            break

        ts = ts + 1

        img = cv2.resize(img, dsize=(frame_width, frame_height))

        h, w, _ = img.shape

        if data_aug:
            if r_f > 5:
                img = cv2.flip(img, 1)
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            if insert_rectangle == 2:
                cv2.rectangle(img, (rh + dx, rw + dy), (rh + dx + ecart_rectangle, rw + dy + ecart_rectangle),
                              (red, green, blue), -1)

            if affine_transform:
                M = cv2.getAffineTransform(pts1, pts2)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if edge_type == 'all':
            blur_kernel = 0
            operator_kernel = 3

        if edge_type == 'all':
            if c_operator == 1:
                img = transform_img_sobel(img, blur_kernel, operator_kernel)
            elif c_operator == 2:
                img = laplace_dev(img, blur_kernel, operator_kernel)
        elif edge_type == 'sobel':
            img = transform_img_sobel(img, blur_kernel, operator_kernel)
        elif edge_type == 'laplace':
            img = laplace_dev(img, blur_kernel, operator_kernel)
        elif edge_type == 'treble':
            img = treble_edges(img, blur_kernel, operator_kernel)
        elif edge_type == 'canny':
            img = cv2.Canny(img, h, w)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = img

        img = img[:, :, [2, 1, 0]]

        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        frames.append(img)
    frames = extract_optical_flow_from_list(frames)
    # print('we will see here %d' %(len(frames)))
    return np.asarray(frames, dtype=np.float32), frames
