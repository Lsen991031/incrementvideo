# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
from random import shuffle
from math import ceil
from scipy import interpolate

class VideoRecord(object):
    def __init__(self, row, index=None, is_ex=False):
        self._data = row
        self._index = index
        self._is_exemplar = is_ex

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def indices(self):
        if self._index is not None:
            return list(self._index.data.numpy())
        else:
            return None

    @property
    def is_exemplar(self):
        return self._is_exemplar


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, task_list, class_indexer,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False,
                 exemplar_list=None, exemplar_only=False,
                 store_frames=None,
                 is_entire=False,
                 exemplar_per_class=5, current_head=101, diverse_rate=False,
                 cl_method=None, age=0, is_img_end=False
                 ):

        self.root_path = root_path
        self.list_file = list_file
        self.task_list = task_list
        self.class_indexer = class_indexer
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation

        self.num_current_class = []
        self.is_img_end = is_img_end
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self.cl_method = cl_method
        self.exemplar_list = exemplar_list
        # print("exemplar_list.device is : {}".format(exemplar_list.device))
        # print("self.exemplar_list.device is : {}".format(self.exemplar_list.device))
        # print("dataset.device is : {}".format(device))
        self.exemplar_only = exemplar_only
        self.store_frames = store_frames
        self.is_entire = is_entire
        self.diverse_rate = diverse_rate

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                # print(self.root_path) # /home/ls/mmaction2-old/data/hmdb51/rawframes
                # print(directory) # wave/A_Beautiful_Mind_2_wave_u_cm_np1_ri_med_3
                # print(self.image_tmpl.format(idx)) # img_00009.jpg
                # print(os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
                
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        # print("tmp.self.task_list shape is {}".format(self.list_file))
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        # print("tmp.self.task_list shape is {}".format(len(tmp)))
        if not self.exemplar_only:
            self.video_list = [VideoRecord(item) for item in tmp if int(item[2]) in self.task_list] # Filter data list with the task list
            print("selg.self.task_list shape is {}".format(self.task_list))
            print("self.video_list.self.task_list shape is {}".format(len(self.video_list)))
            # for item in tmp:
            #     if int(item[2]) in self.task_list:
            #         print(item)
            #         print(ex_i[item[0]][2])

            for i in range(len(self.task_list)):
                # print("task_list[i]selg.self.task_list shape is {}".format(self.task_list[i]))
                v1 = [VideoRecord(item) for item in tmp if int(item[2]) is self.task_list[i]]
                # print(len(v1))
                self.num_current_class.append(len(v1))
            print(self.num_current_class)

            if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                for v in self.video_list:
                    v._data[1] = int(v._data[1]) / 2
            print('video number : {:d}'.format(len(self.video_list)))

            if self.exemplar_list:
                for ex_i in self.exemplar_list:
                    temp_i = [VideoRecord(item,ex_i[item[0]][2],True) for item in tmp if item[0] in ex_i.keys()]
                    # print("temp_i is {}".format(temp_i))
                    # self.video_list = self.video_list + temp_i
                    # temp_i.
                    self.video_list = self.video_list + temp_i
                    # print("{}{}{}{}{}{}{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
            print('video number + exemplar : {:d}'.format(len(self.video_list)))
        else: # only exemplar
            self.video_list = []
            for ex_i in self.exemplar_list:
                # print("88888888888888888888888888888888")
                temp_i = [VideoRecord(item,ex_i[item[0]][2],True) for item in tmp if item[0] in ex_i.keys()]
                # for item in tmp:
                #     if item[0] in ex_i.keys():
                #         print(item)
                #         print(ex_i[item[0]][2])

                # print("88888888888888888888888888888888{}".format(temp_i[0].label))
                # print("88888888888888888888888888888888{}".format(temp_i[1].label))
                self.video_list = self.video_list + temp_i
            print('exemplar : {:d}'.format(len(self.video_list)))

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1

        else:  # normal sample
            if self.diverse_rate:
                num_segments = randint(1,self.num_segments+1,size=1)[0]
            else:
                num_segments = self.num_segments

            average_duration = (record.num_frames - self.new_length + 1) // num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=num_segments)
            elif record.num_frames > num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=num_segments))
            else:
                offsets = np.zeros((num_segments,))
            return offsets + 1

    def _sample_random_indices(self, record):
        indices = np.random.choice(record.num_frames,self.num_segments,replace=False)
        return np.sort(indices) + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            num_segments = self.num_segments
            if record.num_frames > num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
            else:
                offsets = np.zeros((num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1

        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder

        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if self.store_frames == 'random':
            segment_indices = self._sample_random_indices(record)
        elif not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def duplicate_index(self,index):
        x = np.arange(self.num_segments)
        if len(index) == 1:
            index = np.repeat(index,2)
        xp = np.linspace(0,self.num_segments-1,len(index))
        f = interpolate.interp1d(xp,index,kind='nearest')
        index = f(x)
        return index

    def get(self, record, indices):
        if record.indices is not None and record.is_exemplar and not self.is_entire:
            indices = record.indices
        if len(indices) < self.num_segments:
            indices = self.duplicate_index(indices)
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p) # wave/A_Beautiful_Mind_2_wave_u_cm_np1_ri_med_3
                images.extend(seg_imgs) # p == img_00009.jpg
                if p < record.num_frames:
                    p += 1

        indices = np.array(indices).astype(int)
        process_data = self.transform(images)
        indices = np.array(indices).astype(int)

        # print("/homels/hsnet/checkpoint/hmdb51/exemplar")
        img_end = None
        if(record._is_exemplar and self.is_img_end):
            filename = os.path.join("/home/ls/hsnet/checkpoint/hmdb51/exemplar", record.path , 'img.pth')
            img_end = torch.load(filename)
            print(img_end.shape)
            # img_end = img_end.repeat(8,1,1).cpu()
            # print(img_end.shape)
            return img_end, self.class_indexer[record.label], [record.path, record.num_frames, indices.astype(int)]
        # print([record.path, record.num_frames, indices.astype(int)][0])

        # print(process_data.shape)
        # print("class_indexer per class : {}".format(self.class_indexer)) # class_indexer per class : {43: 0, 41: 1, 23: 2, 14: 3, 13: 4, 40: 5, 42: 6, 22: 7, 16: 8, 45: 9, 17: 10, 10: 11, 27: 12, 46: 13, 35: 14, 8: 15, 2: 16, 34: 17, 1: 18, 37: 19, 21: 20, 0: 21, 18: 22, 36: 23, 38: 24, 24: 25, 12: 26, 6: 27, 15: 28, 20: 29, 25: 30, 48: 31, 30: 32, 19: 33, 44: 34, 26: 35, 7: 36, 28: 37, 11: 38, 5: 39, 32: 40, 4: 41, 9: 42, 47: 43, 39: 44, 31: 45, 3: 46, 29: 47, 50: 48, 49: 49, 33: 50}
        # print("self.class_indexer[record.label] is {}".format(record.label))

        return process_data, self.class_indexer[record.label], [record.path, record.num_frames, indices.astype(int)]


    def __len__(self):
        return len(self.video_list)

