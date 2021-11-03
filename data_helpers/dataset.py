from array import array
from builtins import type

import torch
import torch.utils.data as data_utl
import pickle as pkl

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

from random import gauss
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr, \
    scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h


# from data_helpers.flow_generators.flownet2.infer import call_inference, inference

import torch
import torch.utils.data as data
from termcolor import colored

import os, math, random
from os.path import *

import cv2
import time


import argparse

from data_helpers.flow_generators.gma.core.network import RAFTGMA
from data_helpers.flow_generators.gma.core.utils.utils import InputPadder



from data_helpers.frame_loader.load_frames import *
from glob import glob

def choose_24_of_75(label):
    new_label = np.zeros([24])
    j = 0
    for i in label.shape[0]:
        if i % 3 == 0:
            new_label[j] = label[i]
    return


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]


class ImageParser(data.Dataset):
    def __init__(self, args, is_cropped, images=None, iext='png', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        images = images
        self.image_list = []
        for i in range(len(images) - 1):
            im1 = images[i]
            im2 = images[i + 1]
            self.image_list += [[im1, im2]]

        self.size = len(self.image_list)

        self.frame_size = self.image_list[0][0].shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        args.inference_size = self.render_size

    def __getitem__(self, index):

        index = index % self.size

        img1 = self.image_list[index][0]
        img2 = self.image_list[index][1]

        images = [img1, img2]

        # print(type(images))

        images = np.array(images).transpose(3, 0, 1, 2)
        images = torch.from_numpy(images.astype(np.float32))

        # print(images.shape)

        return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

    def __len__(self):
        return self.size * self.replicates


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


correspondance = {'walk': 0, 'handpickup': 1, 'lieonbedthensit': 2,
                  'sitthenstandup': 3, 'crawl': 4, 'lieonbedthenfall': 5,
                  'sitthenfall': 6, 'fall': 7}


def make_dataset(split, data_root, type_input, input_mode, num_classes, nb_frames_per_shot):
    split_file = glob(os.path.join(data_root, '*.json'))[0]
    dataset = []
    # print(os.getcwd() )
    with open(split_file, 'r') as f:
        data = json.load(f)

    with open(split_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    i = 0
    short_data = 0
    for vid in data.keys():
        # print('%s ----- %s'%(data[vid]['subset'], split))
        if data[vid]['subset'] != split:
            continue

        if 'K7' in vid:
            continue

        video_name = data[vid]['oricy']
        # print(video_name)

        file_name = os.path.join(data_root, type_input, video_name)

        # print(file_name)

        if not os.path.exists(file_name):
            print(colored(file_name, 'red'))
            continue

        num_frames = int(data[vid]['duration'])

        if input_mode == 'flow':
            # num_frames = num_frames // 2
            num_frames = num_frames - 1
            if num_frames < nb_frames_per_shot - 1:
                print('num_frames  {:d}      nb_frames_per_shot  {:d}'.format(num_frames, nb_frames_per_shot))
                print("File too small")
                short_data = short_data + 1
                continue
        else:
            if num_frames < nb_frames_per_shot:
                short_data = short_data + 1
                print(file_name)
                print(num_frames)
                print('\n\n')
                continue

        label = np.zeros((num_classes, num_frames), np.float32)

        # fps = num_frames/data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0, num_frames, 1):
                # if fr/fps > ann[1] and fr/fps < ann[2]:
                label[correspondance[ann[0]], fr] = 1  # binary classification
        dataset.append((video_name, label, num_frames))
        i += 1

    if i == 0:
        raise ValueError('----------Unable to load dataset----------\n\n\n')

    return dataset


def transform_flows_frames(flow_numpy):
    nb_frames, _, _, _ = flow_numpy.shape
    flow_frames = []
    for i in range(nb_frames):
        flow_frame = flow_numpy[i, :, :, :]
        flow_frame = flow_frame.transpose([1, 2, 0])
        flow_frame = cv2.resize(flow_frame, (160, 120))
        h, w, c = flow_frame.shape

        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            flow_frame = cv2.resize(flow_frame, dsize=(0, 0), fx=sc, fy=sc)

        flow_frames.append(flow_frame)
    flow_frames = np.asarray(flow_frames, dtype=np.float32)
    # print(flow_frames.shape)
    return flow_frames


def save_numpy_frames(flow_numpy):
    save_flows = 'save_flows'
    ts = time.time()
    now = str(int(ts * 1000000000000))
    if not os.path.exists(save_flows):
        os.makedirs(save_flows)
    if not os.path.exists(os.path.join(save_flows, now)):
        os.makedirs(os.path.join(save_flows, now))

    nb_frames, _, _, _ = flow_numpy.shape
    for i in range(nb_frames):
        mask = np.zeros((120, 160, 3), np.uint8)
        # Sets image saturation to maximum
        mask[..., 1] = 255

        flow_frame = flow_numpy[i, :, :, :]
        flow_frame = flow_frame.transpose([1, 2, 0])
        flow_frame = cv2.resize(flow_frame, (160, 120))
        h, w, c = flow_frame.shape

        magnitude, angle = cv2.cartToPolar(flow_frame[..., 0], flow_frame[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(save_flows, now, str(i).zfill(6)) + '.png', rgb)


def get_flow_model():

    
    flags = argparse.Namespace()
    flags.model = 'pretrained/gma-sintel.pth'
    flags.model_name = 'GMA'
    flags.path = 'imgs'
    flags.num_heads = 1
    flags.position_only = False
    flags.position_and_content = False
    flags.mixed_precision = False

    model = torch.nn.DataParallel(RAFTGMA(flags))
    model.load_state_dict(torch.load(flags.model))
    # print(f"Loaded checkpoint at {flags.model}")

    model = model.module
    model.cuda()
    model.eval()

    return model


class Video_Datasets(data_utl.Dataset):

    def __init__(self, data_root,  split, source_target, flags, domain_key, is_train, modality,  transforms=None):

        # print("prepare dataset cme")
        self.is_train = is_train
        self.data = make_dataset(split= split, data_root= os.path.join(data_root, source_target), type_input=flags.input ,input_mode=flags.input_mode, num_classes= flags.num_classes,nb_frames_per_shot= flags.nb_frames)
        self.args = flags
        self.flow_model = get_flow_model()
        self.domain_key = domain_key
        self.nb_frames = flags.nb_frames
        self.transforms = transforms
        self.input_mode = flags.input_mode
        self.middle_transform = flags.middle_transform
        self.data_root = os.path.join(data_root, source_target)
        self.modality = modality
        self.video_augmentations = flags.video_augmentations
        self.affine_transform = flags.affine_transform
        self.taille = len(self.data)

        # print("dataset prepared cme")

    def get_video(self, index):
        vid, label, total_number_frames = self.data[index]
        if self.input_mode == 'flow':
            start_f = 0 if total_number_frames == (self.nb_frames - 1) else random.randint(0,
                                                                                           total_number_frames - self.nb_frames)
        else:
            start_f = 0 if total_number_frames == self.nb_frames else random.randint(0, total_number_frames - (
                    self.nb_frames + 1))

        # print('enfin %d' %(total_number_frames-start_f))
        numpy_imgs, list_imgs = load_frames(self.data_root, vid, start_f, self.nb_frames, self.video_augmentations,
                                      affine_transform=self.affine_transform,
                                      domain_key=self.domain_key,
                                      is_train=self.is_train,
                                      modality=self.modality,
                                      verbose=False, original_mode=self.input_mode,
                                      middle_transform=self.middle_transform,
                                      frame_height=192,
                                      frame_width=256
                                      )

        # print('get_video')
        # print(imgs.shape)

        if self.args.middle_transform == 'flow':
            flows = []
            for it in range(len(list_imgs) - 1):
                image1 = list_imgs[it]
                image2 = list_imgs[it + 1]
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = self.flow_model(image1, image2, iters=12, test_mode=True)
                flow_up = flow_up.cpu().numpy()
                flows.append(flow_up)


            save_numpy_frames(flows)
            flows_cpu = transform_flows_frames(flows)
            flows_cpu = self.transforms(flows_cpu)
            frames_pt = torch.from_numpy(flows_cpu.transpose([3, 0, 1, 2]))
            label = label[:, start_f:start_f + self.nb_frames - 1]
        else:
            numpy_imgs = self.transforms(numpy_imgs)
            frames_pt = video_to_tensor(numpy_imgs)
            label = label[:, start_f:start_f + self.nb_frames]

        return frames_pt, torch.from_numpy(label).squeeze(), torch.tensor(
            self.domain).long().squeeze()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        video, label, domain = self.get_video(index)

        return video, label, domain


    def __len__(self):
        return len(self.data)

