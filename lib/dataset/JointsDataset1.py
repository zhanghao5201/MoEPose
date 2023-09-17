# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

import os
import json_tricks as json
import torchvision
import numpy as np
import cv2
import torch
import copy
from PIL import Image
import math
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from collections import OrderedDict
import logging
import os
logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        image_file_new = db_rec['image_new']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            data_numpy_new=save_image1(s, joints, joints_vis,data_numpy)
            #data_numpy_new = zipreader.imread(
            #    image_file_new, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            #)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )            
            data_numpy_new=save_image1(s, joints, joints_vis,data_numpy)
            #cv2.imwrite("./zh2.jpg", data_numpy)
            #cv2.imwrite("./zh3{}.jpg".format(image_file.split('/')[-1].split('.')[0]), data_numpy_new)
            #print("./zh3{}.jpg".format(image_file.split('/')[-1].split('.')[0]))
            #data_numpy_new = cv2.imread(
            #    image_file_new, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            #)

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            data_numpy_new = cv2.cvtColor(data_numpy_new, cv2.COLOR_BGR2RGB)
            #cv2.imwrite("./zh.jpg", data_numpy)
            #cv2.imwrite("./zh1.jpg", data_numpy_new)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))



        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                data_numpy_new = data_numpy_new[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        input_new = cv2.warpAffine(
            data_numpy_new,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)
            input_new = self.transform(input_new)
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input,input_new,target, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
        
def save_image1(scale, batch_joints, batch_joints_vis, imageys):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    #print("zh1",batch_joints[11].shape,batch_joints[0],scale.shape,scale)
    #print("zh",batch_joints.shape,batch_joints[0].shape,batch_joints[0],scale.shape,scale)
    batch_joints=batch_joints[:,0:2]
    #print(batch_joints.shape,batch_joints)
    x = random.randint(0, 1)
    y = random.randint(0, 1)
    z = random.randint(1, 3)
    img4=imageys
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    img4 = Image.fromarray(img4)
    img4 = img4.convert('RGBA')
    #img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)
    
    if x == 0 and y == 0 and z == 1:
        n1 = random.randint(1, 2584)
        kuai_name1 = os.path.join('gengxin3', "{}-14".format(n1) + '.png')
        img1 = Image.open(kuai_name1)  
        n2 = random.randint(1, 2339)
        n4 = random.randint(1, 8583)
        if random.randint(0, 1):
            kuai_name2 = os.path.join('gengxin3', "{}-20".format(n2) + '.png')
        else:
            kuai_name2 = os.path.join('gengxin3', "{}-23".format(n4) + '.png')
        img2 = Image.open(kuai_name2) 
        n3 = random.randint(1, 9630)
        kuai_name3 = os.path.join('gengxin3', "{}-24".format(n3) + '.png')
        img3 = Image.open(kuai_name3)  

        w1 = img1.size[0]
        h1 = img1.size[1]
        w2 = img2.size[0]
        h2 = img2.size[1]
        w3 = img3.size[0]
        h3 = img3.size[1]
          

        if batch_joints[11][0] != 0 and batch_joints[15][0] != 0 and (
                abs(batch_joints[2] - batch_joints[0]) > scale * 40).all():
            ss = abs(batch_joints[2] - batch_joints[0])
        else:
            ss = scale * 200 * 0.5 * 0.8
        if w1 > h1:
            if w1 < ss[1] * 0.3:
                w11 = ss[1] * 0.4
            else:
                w11 = ss[1] * 0.6
            h11 = h1 / w1 * w11
        else:
            if h1 < ss[1] * 0.3:
                h11 = ss[1] * 0.4
            else:
                h11 = ss[1] * 0.6
            w11 = w1 / h1 * h11
        img1 = img1.resize((int(w11), int(h11)))
        if batch_joints_vis[11][0] > 0:
            ww = int(batch_joints[11][0] - w11 / 2)
            hh = int(batch_joints[11][1] - h11 / 2 * 0.5)
            
            img4.paste(img1, (ww, hh), img1)
            #img4.save(image_file, format='png')

        if w2 > h2:
            if w2 < ss[1] * 0.3:
                w22 = ss[1] * 0.5
            else:
                w22 = ss[1] * 0.8
            h22 = h2 / w2 * w22
        else:
            if h2 < ss[1] * 0.3:
                h22 = ss[1] * 0.5
            else:
                h22 = ss[1] * 0.8
            w22 = w2 / h2 * h22
        img2 = img2.resize((int(w22), int(h22)))
        if batch_joints_vis[4][0] > 0:
            ww = int(batch_joints[4][0] - w22 / 2)
            hh = int(batch_joints[4][1] - h22 / 2 * 0.5)
            
            img4.paste(img2, (ww, hh), img2)
            #img4.save(image_file, format='png')

        if w3 > h3:
            if w3 < ss[1] * 0.3:
                w33 = ss[1] * 0.5
            else:
                w33 = ss[1]
            h33 = h3 / w3 * w33
        else:
            if h3 < ss[1] * 0.3:
                h33 = ss[1] * 0.5
            else:
                h33 = ss[1]
            w33 = w3 / h3 * h33
        img3 = img3.resize((int(w33), int(h33)))
        if batch_joints_vis[14][0] > 0:
            ww = int(batch_joints[14][0] - w33 / 2)
            hh = int(batch_joints[14][1] - h33 / 2 * 0.5)
            img4.paste(img3, (ww, hh), img3)
            #img4.save(image_file, format='png')


    if x == 0 and y == 0 and z == 3:
        n1 = random.randint(1, 2584)
        kuai_name1 = os.path.join('gengxin3', "{}-14".format(n1) + '.png')
        img1 = Image.open(kuai_name1)  
        n2 = random.randint(1, 2339)
        n4 = random.randint(1, 8583)
        if random.randint(0, 1):
            kuai_name2 = os.path.join('gengxin3', "{}-20".format(n2) + '.png')
        else:
            kuai_name2 = os.path.join('gengxin3', "{}-23".format(n4) + '.png')
        img2 = Image.open(kuai_name2)  
        n3 = random.randint(1, 224)
        kuai_name3 = os.path.join('gengxin3', "{}-10".format(n3) + '.png')
        img3 = Image.open(kuai_name3)  

        w1 = img1.size[0]
        h1 = img1.size[1]
        w2 = img2.size[0]
        h2 = img2.size[1]
        w3 = img3.size[0]
        h3 = img3.size[1]  

        if batch_joints[11][0] != 0 and batch_joints[15][0] != 0 and (
                abs(batch_joints[2] - batch_joints[0]) > scale * 40).all():
            ss = abs(batch_joints[2] - batch_joints[0])
        else:
            ss = scale * 200 * 0.5 * 0.8

        if w1 > h1:
            if w1 < ss[1] * 0.3:
                w11 = ss[1] * 0.4
            else:
                w11 = ss[1] * 0.6
            h11 = h1 / w1 * w11
        else:
            if h1 < ss[1] * 0.3:
                h11 = ss[1] * 0.4
            else:
                h11 = ss[1] * 0.6
            w11 = w1 / h1 * h11
        img1 = img1.resize((int(w11), int(h11)))
        if batch_joints_vis[11][0] > 0:
            ww = int(batch_joints[11][0] - w11 / 2)
            hh = int(batch_joints[11][1] - h11 / 2 * 0.5)            
            img4.paste(img1, (ww, hh), img1)
            #img4.save(image_file, format='png')

        if w2 > h2:
            if w2 < ss[1] * 0.3:
                w22 = ss[1] * 0.5
            else:
                w22 = ss[1] * 0.8
            h22 = h2 / w2 * w22
        else:
            if h2 < ss[1] * 0.3:
                h22 = ss[1] * 0.5
            else:
                h22 = ss[1] * 0.8
            w22 = w2 / h2 * h22
        img2 = img2.resize((int(w22), int(h22)))
        if batch_joints_vis[4][0] > 0:
            ww = int(batch_joints[4][0] - w22 / 2)
            hh = int(batch_joints[4][1] - h22 / 2 * 0.5)
            img4.paste(img2, (ww, hh), img2)
            #img4.save(image_file, format='png')

        if w3 > h3:
            if w3 < ss[1] * 0.3:
                w33 = ss[1] * 0.5
            else:
                w33 = ss[1]
            h33 = h3 / w3 * w33
        else:
            if h3 < ss[1] * 0.3:
                h33 = ss[1] * 0.5
            else:
                h33 = ss[1]
            w33 = w3 / h3 * h33
        img3 = img3.resize((int(w33), int(h33)))
        if batch_joints_vis[7][0] > 0:
            ww = int(batch_joints[7][0]- w33 / 2)
            hh = int(batch_joints[7][1]  + h33 / 2 * 0.5)
            img4.paste(img3, (ww, hh), img3)
            #img4.save(image_file, format='png')

    if x == 0 and y == 1 and z == 1:
        n1 = random.randint(1, 2584)
        kuai_name1 = os.path.join('gengxin3', "{}-14".format(n1) + '.png')
        img1 = Image.open(kuai_name1)  
        n2 = random.randint(1, 2367)
        n4 = random.randint(1, 8583)
        if random.randint(0, 1):
            kuai_name2 = os.path.join('gengxin3', "{}-21".format(n2) + '.png')
        else:
            kuai_name2 = os.path.join('gengxin3', "{}-23".format(n4) + '.png')
        img2 = Image.open(kuai_name2)  
        n3 = random.randint(1, 9630)
        kuai_name3 = os.path.join('gengxin3', "{}-24".format(n3) + '.png')
        img3 = Image.open(kuai_name3)  

        w1 = img1.size[0]
        h1 = img1.size[1]
        w2 = img2.size[0]
        h2 = img2.size[1]
        w3 = img3.size[0]
        h3 = img3.size[1]

        if batch_joints[11][0] != 0 and batch_joints[15][0] != 0 and (
                abs(batch_joints[2] - batch_joints[0]) > scale * 40).all():
            ss = abs(batch_joints[2] - batch_joints[0])
        else:
            ss = scale * 200 * 0.5 * 0.8

        if w1 > h1:
            if w1 < ss[1] * 0.3:
                w11 = ss[1] * 0.4
            else:
                w11 = ss[1] * 0.6
            h11 = h1 / w1 * w11
        else:
            if h1 < ss[1] * 0.3:
                h11 = ss[1] * 0.4
            else:
                h11 = ss[1] * 0.6
            w11 = w1 / h1 * h11
        img1 = img1.resize((int(w11), int(h11)))
        if batch_joints_vis[11][0] > 0:
            ww = int(batch_joints[11][0] - w11 / 2)
            hh = int(batch_joints[11][1] - h11 / 2 * 0.5)
            img4.paste(img1, (ww, hh), img1)
            #img4.save(image_file, format='png')

        if w2 > h2:
            if w2 < ss[1] * 0.3:
                w22 = ss[1] * 0.5
            else:
                w22 = ss[1] * 0.8
            h22 = h2 / w2 * w22
        else:
            if h2 < ss[1] * 0.3:
                h22 = ss[1] * 0.5
            else:
                h22 = ss[1] * 0.8
            w22 = w2 / h2 * h22
        img2 = img2.resize((int(w22), int(h22)))
        if batch_joints_vis[1][0] > 0:
            ww = int(batch_joints[1][0] - w22 / 2)
            hh = int(batch_joints[1][1] - h22 / 2 * 0.5)
            img4.paste(img2, (ww, hh), img2)
            #img4.save(image_file, format='png')

        if w3 > h3:
            if w3 < ss[1] * 0.3:
                w33 = ss[1] * 0.5
            else:
                w33 = ss[1]
            h33 = h3 / w3 * w33
        else:
            if h3 < ss[1] * 0.3:
                h33 = ss[1] * 0.5
            else:
                h33 = ss[1]
            w33 = w3 / h3 * h33
        img3 = img3.resize((int(w33), int(h33)))
        if batch_joints_vis[14][0] > 0:
            ww = int(batch_joints[14][0] - w33 / 2)
            hh = int(batch_joints[14][1] - h33 / 2 * 0.5)
            img4.paste(img3, (ww, hh), img3)
            #img4.save(image_file, format='png')


    if x == 0 and y == 1 and z == 3:
        n1 = random.randint(1, 2584)
        kuai_name1 = os.path.join('gengxin3', "{}-14".format(n1) + '.png')
        img1 = Image.open(kuai_name1)  
        n2 = random.randint(1, 2367)
        n4 = random.randint(1, 8583)
        if random.randint(0, 1):
            kuai_name2 = os.path.join('gengxin3', "{}-21".format(n2) + '.png')
        else:
            kuai_name2 = os.path.join('gengxin3', "{}-23".format(n4) + '.png')
        img2 = Image.open(kuai_name2)  
        n3 = random.randint(1, 224)
        kuai_name3 = os.path.join('gengxin3', "{}-10".format(n3) + '.png')
        img3 = Image.open(kuai_name3) 

        w1 = img1.size[0]
        h1 = img1.size[1]
        w2 = img2.size[0]
        h2 = img2.size[1]
        w3 = img3.size[0]
        h3 = img3.size[1]
        if batch_joints[11][0] != 0 and batch_joints[15][0] != 0 and (
                abs(batch_joints[2] - batch_joints[0]) > scale * 40).all():
            ss = abs(batch_joints[2] - batch_joints[0])
        else:
            ss = scale * 200 * 0.5 * 0.8
        if w1 > h1:
            if w1 < ss[1] * 0.3:
                w11 = ss[1] * 0.4
            else:
                w11 = ss[1] * 0.6
            h11 = h1 / w1 * w11
        else:
            if h1 < ss[1] * 0.3:
                h11 = ss[1] * 0.4
            else:
                h11 = ss[1] * 0.6
            w11 = w1 / h1 * h11
        img1 = img1.resize((int(w11), int(h11)))
        if batch_joints_vis[11][0] > 0:
            ww = int(batch_joints[11][0] - w11 / 2)
            hh = int(batch_joints[11][1] - h11 / 2 * 0.5)
            img4.paste(img1, (ww, hh), img1)
            #img4.save(image_file, format='png')

        if w2 > h2:
            if w2 < ss[1] * 0.3:
                w22 = ss[1] * 0.5
            else:
                w22 = ss[1] * 0.8
            h22 = h2 / w2 * w22
        else:
            if h2 < ss[1] * 0.3:
                h22 = ss[1] * 0.5
            else:
                h22 = ss[1] * 0.8
            w22 = w2 / h2 * h22
        img2 = img2.resize((int(w22), int(h22)))
        if batch_joints_vis[1][0] > 0:
            ww = int(batch_joints[1][0] - w22 / 2)
            hh = int(batch_joints[1][1] - h22 / 2 * 0.5)
            img4.paste(img2, (ww, hh), img2)
            #img4.save(image_file, format='png')

        if w3 > h3:
            if w3 < ss[1] * 0.3:
                w33 = ss[1] * 0.5
            else:
                w33 = ss[1]
            h33 = h3 / w3 * w33
        else:
            if h3 < ss[1] * 0.3:
                h33 = ss[1] * 0.5
            else:
                h33 = ss[1]
            w33 = w3 / h3 * h33
        img3 = img3.resize((int(w33), int(h33)))
        if batch_joints_vis[7][0] > 0:
            ww = int(batch_joints[7][0]- w33 / 2)
            hh = int(batch_joints[7][1] + h33 / 2 * 0.5)
            img4.paste(img3, (ww, hh), img3)
            #img4.save(image_file, format='png')
        

    if x == 1 and y == 0 and z == 1:
        n1 = random.randint(1, 2938)
        kuai_name1 = os.path.join('gengxin3', "{}-15".format(n1) + '.png')
        img1 = Image.open(kuai_name1)  
        n2 = random.randint(1, 2339)
        n4 = random.randint(1, 8583)
        if random.randint(0, 1):
            kuai_name2 = os.path.join('gengxin3', "{}-20".format(n2) + '.png')
        else:
            kuai_name2 = os.path.join('gengxin3', "{}-23".format(n4) + '.png')
        img2 = Image.open(kuai_name2)  
        n3 = random.randint(1, 9630)
        kuai_name3 = os.path.join('gengxin3', "{}-24".format(n3) + '.png')
        img3 = Image.open(kuai_name3)  

        w1 = img1.size[0]
        h1 = img1.size[1]
        w2 = img2.size[0]
        h2 = img2.size[1]
        w3 = img3.size[0]
        h3 = img3.size[1] 

        if batch_joints[11][0] != 0 and batch_joints[15][0] != 0 and (
                abs(batch_joints[2] - batch_joints[0]) > scale * 40).all():
            ss = abs(batch_joints[2] - batch_joints[0])
        else:
            ss = scale * 200 * 0.5 * 0.8

        if w1 > h1:
            if w1 < ss[1] * 0.3:
                w11 = ss[1] * 0.4
            else:
                w11 = ss[1] * 0.6
            h11 = h1 / w1 * w11
        else:
            if h1 < ss[1] * 0.3:
                h11 = ss[1] * 0.4
            else:
                h11 = ss[1] * 0.6
            w11 = w1 / h1 * h11
        img1 = img1.resize((int(w11), int(h11)))
        if batch_joints_vis[14][0] > 0:
            ww = int(batch_joints[14][0] - w11 / 2)
            hh = int(batch_joints[14][1] - h11 / 2 * 0.5)
            img4.paste(img1, (ww, hh), img1)
            #img4.save(image_file, format='png')

        if w2 > h2:
            if w2 < ss[1] * 0.3:
                w22 = ss[1] * 0.5
            else:
                w22 = ss[1] * 0.8
            h22 = h2 / w2 * w22
        else:
            if h2 < ss[1] * 0.3:
                h22 = ss[1] * 0.5
            else:
                h22 = ss[1] * 0.8
            w22 = w2 / h2 * h22
        img2 = img2.resize((int(w22), int(h22)))
        if batch_joints_vis[4][0] > 0:
            ww = int(batch_joints[4][0] - w22 / 2)
            hh = int(batch_joints[4][1] - h22 / 2 * 0.5)
            img4.paste(img2, (ww, hh), img2)
            #img4.save(image_file, format='png')

        if w3 > h3:
            if w3 < ss[1] * 0.3:
                w33 = ss[1] * 0.5
            else:
                w33 = ss[1]
            h33 = h3 / w3 * w33
        else:
            if h3 < ss[1] * 0.3:
                h33 = ss[1] * 0.5
            else:
                h33 = ss[1]
            w33 = w3 / h3 * h33
        img3 = img3.resize((int(w33), int(h33)))
        if batch_joints_vis[11][0] > 0:
            ww = int(batch_joints[11][0] - w33 / 2)
            hh = int(batch_joints[11][1] - h33 / 2 * 0.5)
            img4.paste(img3, (ww, hh), img3)
            #img4.save(image_file, format='png')

    if x == 1 and y == 0 and z == 3:
        n1 = random.randint(1, 2938)
        kuai_name1 = os.path.join('gengxin3', "{}-15".format(n1) + '.png')
        img1 = Image.open(kuai_name1)  
        n2 = random.randint(1, 2339)
        n4 = random.randint(1, 8583)
        if random.randint(0, 1):
            kuai_name2 = os.path.join('gengxin3', "{}-20".format(n2) + '.png')
        else:
            kuai_name2 = os.path.join('gengxin3', "{}-23".format(n4) + '.png')
        img2 = Image.open(kuai_name2)  
        n3 = random.randint(1, 224)
        kuai_name3 = os.path.join('gengxin3', "{}-10".format(n3) + '.png')
        img3 = Image.open(kuai_name3) 

        w1 = img1.size[0]
        h1 = img1.size[1]
        w2 = img2.size[0]
        h2 = img2.size[1]
        w3 = img3.size[0]
        h3 = img3.size[1]
        if batch_joints[11][0] != 0 and batch_joints[15][0] != 0 and (
                abs(batch_joints[2] - batch_joints[0]) > scale * 40).all():
            ss = abs(batch_joints[2] - batch_joints[0])
        else:
            ss = scale * 200 * 0.5 * 0.8

        if w1 > h1:
            if w1 < ss[1] * 0.3:
                w11 = ss[1] * 0.4
            else:
                w11 = ss[1] * 0.6
            h11 = h1 / w1 * w11
        else:
            if h1 < ss[1] * 0.3:
                h11 = ss[1] * 0.4
            else:
                h11 = ss[1] * 0.6
            w11 = w1 / h1 * h11
        img1 = img1.resize((int(w11), int(h11)))
        if batch_joints_vis[14][0] > 0:
            ww = int(batch_joints[14][0] - w11 / 2)
            hh = int(batch_joints[14][1] - h11 / 2 * 0.5)
            img4.paste(img1, (ww, hh), img1)
            #img4.save(image_file, format='png')

        if w2 > h2:
            if w2 < ss[1] * 0.3:
                w22 = ss[1] * 0.5
            else:
                w22 = ss[1] * 0.8
            h22 = h2 / w2 * w22
        else:
            if h2 < ss[1] * 0.3:
                h22 = ss[1] * 0.5
            else:
                h22 = ss[1] * 0.8
            w22 = w2 / h2 * h22
        img2 = img2.resize((int(w22), int(h22)))
        if batch_joints_vis[4][0] > 0:
            ww = int(batch_joints[4][0] - w22 / 2)
            hh = int(batch_joints[4][1] - h22 / 2 * 0.5)
            img4.paste(img2, (ww, hh), img2)
            #img4.save(image_file, format='png')

        if w3 > h3:
            if w3 < ss[1] * 0.3:
                w33 = ss[1] * 0.5
            else:
                w33 = ss[1]
            h33 = h3 / w3 * w33
        else:
            if h3 < ss[1] * 0.3:
                h33 = ss[1] * 0.5
            else:
                h33 = ss[1]
            w33 = w3 / h3 * h33
        img3 = img3.resize((int(w33), int(h33)))
        if batch_joints_vis[7][0] > 0:
            ww = int(batch_joints[7][0] - w33 / 2)
            hh = int(batch_joints[7][1]  + h33 / 2 * 0.5)
            img4.paste(img3, (ww, hh), img3)
            #img4.save(image_file, format='png')
    if x == 1 and y == 1 and z == 1:
        n1 = random.randint(1, 2938)
        kuai_name1 = os.path.join('gengxin3', "{}-15".format(n1) + '.png')
        img1 = Image.open(kuai_name1)  
        n2 = random.randint(1, 2367)
        n4 = random.randint(1, 8583)
        if random.randint(0, 1):
            kuai_name2 = os.path.join('gengxin3', "{}-21".format(n2) + '.png')
        else:
            kuai_name2 = os.path.join('gengxin3', "{}-23".format(n4) + '.png')
        img2 = Image.open(kuai_name2)  
        n3 = random.randint(1, 9630)
        kuai_name3 = os.path.join('gengxin3', "{}-24".format(n3) + '.png')
        img3 = Image.open(kuai_name3)  

        w1 = img1.size[0]
        h1 = img1.size[1]
        w2 = img2.size[0]
        h2 = img2.size[1]
        w3 = img3.size[0]
        h3 = img3.size[1]

        if batch_joints[11][0] != 0 and batch_joints[15][0] != 0 and (
                abs(batch_joints[2] - batch_joints[0]) > scale * 40).all():
            ss = abs(batch_joints[2] - batch_joints[0])
        else:
            ss = scale * 200 * 0.5 * 0.8
        if w1 > h1:
            if w1 < ss[1] * 0.3:
                w11 = ss[1] * 0.4
            else:
                w11 = ss[1] * 0.6
            h11 = h1 / w1 * w11
        else:
            if h1 < ss[1] * 0.3:
                h11 = ss[1] * 0.4
            else:
                h11 = ss[1] * 0.6
            w11 = w1 / h1 * h11
        img1 = img1.resize((int(w11), int(h11)))
        if batch_joints_vis[14][0] > 0:
            ww = int(batch_joints[14][0] - w11 / 2)
            hh = int(batch_joints[14][1] - h11 / 2 * 0.5)
            img4.paste(img1, (ww, hh), img1)
            #img4.save(image_file, format='png')

        if w2 > h2:
            if w2 < ss[1] * 0.3:
                w22 = ss[1] * 0.5
            else:
                w22 = ss[1] * 0.8
            h22 = h2 / w2 * w22
        else:
            if h2 < ss[1] * 0.3:
                h22 = ss[1] * 0.5
            else:
                h22 = ss[1] * 0.8
            w22 = w2 / h2 * h22
        img2 = img2.resize((int(w22), int(h22)))
        if batch_joints_vis[1][0] > 0:
            ww = int(batch_joints[1][0] - w22 / 2)
            hh = int(batch_joints[1][1] - h22 / 2 * 0.5)
            img4.paste(img2, (ww, hh), img2)
            #img4.save(image_file, format='png')

        if w3 > h3:
            if w3 < ss[1] * 0.3:
                w33 = ss[1] * 0.5
            else:
                w33 = ss[1]
            h33 = h3 / w3 * w33
        else:
            if h3 < ss[1] * 0.3:
                h33 = ss[1] * 0.5
            else:
                h33 = ss[1]
            w33 = w3 / h3 * h33
        img3 = img3.resize((int(w33), int(h33)))
        if batch_joints_vis[11][0] > 0:
            ww = int(batch_joints[11][0] - w33 / 2)
            hh = int(batch_joints[11][1] - h33 / 2 * 0.5)
            img4.paste(img3, (ww, hh), img3)
            #img4.save(image_file, format='png')


    if x == 1 and y == 1 and z == 3:
        n1 = random.randint(1, 2938)
        kuai_name1 = os.path.join('gengxin3', "{}-15".format(n1) + '.png')
        img1 = Image.open(kuai_name1)  
        n2 = random.randint(1, 2367)
        n4 = random.randint(1, 8583)
        if random.randint(0, 1):
            kuai_name2 = os.path.join('gengxin3', "{}-21".format(n2) + '.png')
        else:
            kuai_name2 = os.path.join('gengxin3', "{}-23".format(n4) + '.png')
        img2 = Image.open(kuai_name2)  
        n3 = random.randint(1, 224)
        kuai_name3 = os.path.join('gengxin3', "{}-10".format(n3) + '.png')
        img3 = Image.open(kuai_name3)  

        w1 = img1.size[0]
        h1 = img1.size[1]
        w2 = img2.size[0]
        h2 = img2.size[1]
        w3 = img3.size[0]
        h3 = img3.size[1]

        if batch_joints[11][0] != 0 and batch_joints[15][0] != 0 and (
                abs(batch_joints[2] - batch_joints[0]) > scale * 40).all():
            ss = abs(batch_joints[2] - batch_joints[0])
        else:
            ss = scale * 200 * 0.5 * 0.8

        if w1 > h1:
            if w1 < ss[1] * 0.3:
                w11 = ss[1] * 0.4
            else:
                w11 = ss[1] * 0.6
            h11 = h1 / w1 * w11
        else:
            if h1 < ss[1] * 0.3:
                h11 = ss[1] * 0.4
            else:
                h11 = ss[1] * 0.6
            w11 = w1 / h1 * h11
        img1 = img1.resize((int(w11), int(h11)))
        if batch_joints_vis[14][0] > 0:
            ww = int(batch_joints[14][0] - w11 / 2)
            hh = int(batch_joints[14][1] - h11 / 2 * 0.5)
            img4.paste(img1, (ww, hh), img1)
            #img4.save(image_file, format='png')

        if w2 > h2:
            if w2 < ss[1] * 0.3:
                w22 = ss[1] * 0.5
            else:
                w22 = ss[1] * 0.8
            h22 = h2 / w2 * w22
        else:
            if h2 < ss[1] * 0.3:
                h22 = ss[1] * 0.5
            else:
                h22 = ss[1] * 0.8
            w22 = w2 / h2 * h22
        img2 = img2.resize((int(w22), int(h22)))
        if batch_joints_vis[1][0] > 0:
            ww = int(batch_joints[1][0] - w22 / 2)
            hh = int(batch_joints[1][1] - h22 / 2 * 0.5)
            img4.paste(img2, (ww, hh), img2)
            #img4.save(image_file, format='png')

        if w3 > h3:
            if w3 < ss[1] * 0.3:
                w33 = ss[1] * 0.5
            else:
                w33 = ss[1]
            h33 = h3 / w3 * w33
        else:
            if h3 < ss[1] * 0.3:
                h33 = ss[1] * 0.5
            else:
                h33 = ss[1]
            w33 = w3 / h3 * h33
        img3 = img3.resize((int(w33), int(h33)))
        if batch_joints_vis[7][0] > 0:
            ww = int(batch_joints[7][0]- w33 / 2)
            hh = int(batch_joints[7][1]  + h33 / 2 * 0.5)
            img4.paste(img3, (ww, hh), img3)
            #img4.save(image_file, format='png')
    #print("zh221",type(img4))
    #img4 = Image.fromarray(img4)    
    img4 = img4.convert('RGB')
    img4 = np.asarray(img4)
    img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)
    return img4

