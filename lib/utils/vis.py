# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2

from core.inference import get_max_preds

class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(self.color[i])

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(point_color[i])


# (R,G,B)
color3 = [(243,176,252),(243,176,252),(243,176,252),
    (240,176,0), (240,176,0), (240,176,0),
    (127,2,240),(127,2,240),(255,0,0),
    (0,255,255), (0,255,255),(0,255,255),
    (142, 209, 169),(142, 209, 169),(142, 209, 169)]

link_pairs3 = [
        [5, 4], [4, 3], [3,6],[6,2],[2,1],[1,0],
        [6, 7], [7, 8], [8, 9],
        [7, 13],[13, 14],[14, 15],
        [7,12], [12,11], [11,10]
        ]
point_color3 = [(240,176,0),(240,176,0),(240,176,0),
            (243,176,252), (243,176,252),(243,176,252),
            (127,2,240),(127,2,240),(255,0,0),(255,0,0),
            (142, 209, 169),(142, 209, 169),(142, 209, 169),
            (0,255,255),(0,255,255),(0,255,255)]


zhanghao_style = ColorStyle(color3, link_pairs3, point_color3)


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict
def save_batch_image_test(batch_image, batch_joints, batch_joints_vis, file_name, link_pairs,ring_color,nrow=8, padding=2,):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)#鐢诲浘锛宯row姣忚鏄惧紡鐨勫浘鐗囨暟
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()#浜ゆ崲缁村害
    ndarr = ndarr.copy()
    ndarr=cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)#涓€琛屽嚑寮犲浘鐗?    
    ymaps = int(math.ceil(float(nmaps) / xmaps))#鏈夊嚑琛?    
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    h=0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break#璺冲嚭for寰幆
            joints = batch_joints[k]
            joints_dict = map_joint_dict(joints)
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, ring_color[h], 2)
                    #cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [0,255,0], 2)
                h=h+1
                if h==16:
                    h=0
            for i, link_pair in enumerate(link_pairs):
                lw = cv2.LINE_4
                if joints_vis[link_pair[0]][0] and joints_vis[link_pair[1]][0] :
                    cv2.line(ndarr,(joints_dict[link_pair[0]][0],joints_dict[link_pair[0]][1]),(joints_dict[link_pair[1]][0],joints_dict[link_pair[1]][1]),link_pair[2],3,lw)
            k=k+1
    cv2.imwrite(file_name, ndarr)
def save_batch_image_with_joints1(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    ndarr=cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    ndarr=cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]#绗嚑涓叧鑺傜殑heatmap
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)#浼壊褰? 
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)#鍔犱簡浼壊褰╃殑鍐嶅姞涓€娆?
            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image#鏈€鍚庣敾鍘熷浘

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return
    colorstyle =zhanghao_style

    #colorstyle.ring_color=np.array(colorstyle.ring_color[0],colorstyle.ring_color[1],colorstyle.ring_color[2])
    #print(colorstyle.ring_color.type)
    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints1(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_test(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt1.jpg'.format(prefix), colorstyle.link_pairs, colorstyle.ring_color
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_test(input, joints_pred,meta['joints_vis'], '{}_pred1.jpg'.format(prefix),colorstyle.link_pairs,colorstyle.ring_color)
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
