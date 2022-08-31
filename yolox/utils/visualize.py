#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["vis", "plot_tracking", "PlotTraj"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


class PlotTraj():
    """
        class to save trajectory and plot traj.
    """
    def __init__(self, track_classes=None, cls_names=None):
        self.track_classes = track_classes
        self.cls_names = cls_names
        self.traj = dict(list())
        self.stats = dict()  # static, moving, turning around

    def update(self, image, tlwhs, obj_ids, frame_id=None):
        image = np.ascontiguousarray(np.copy(image))
        centers = [item[:2] + item[2:] / 2 for item in tlwhs]
        for obj_id, center in zip(obj_ids, centers):
            if obj_id in self.traj.keys():
                self.traj[obj_id].append(center)
            else:
                self.traj[obj_id] = [center]

        self.stats_analysis()
        # print(self.stats)

        # draw line
        for obj_id in self.traj.keys():
            # color = (0, 255, 255) if self.stats[obj_id] == 'turning around' else get_color(abs(obj_id))
            color = get_color(abs(obj_id))
            for i in range(len(self.traj[obj_id]) - 1):
                pt1, pt2 = self.traj[obj_id][i], self.traj[obj_id][i+1]
                cv2.line(image, pt1.astype(int), pt2.astype(int), color, 1, lineType=cv2.LINE_AA)

        return image

    def stats_analysis(self):
        for obj_id in self.traj.keys():
            self.stats[obj_id] = 'static'
            points = self.traj[obj_id].copy()
            pixel_dist = [(points[-1] - p)**2 for p in points]
            pixel_dist = [item.sum()**0.5 for item in pixel_dist]
            if max(pixel_dist[-30:]) < 25:
                continue

            self.stats[obj_id] = 'moving'
            if len(points) > 30:
                pt1, pt2, pt3, pt4 = points[0], points[15], points[-15], points[-1]
                # dif between current angle and initial angle
                dif_theta = self.get_theta_dif(pt1, pt2, pt3, pt4)
                if dif_theta > 0.6*np.pi:
                    self.stats[obj_id] = 'turning around'

    @staticmethod
    def get_theta_dif(p1, p2, q1, q2):
        dif_1, dif_2 = p2 - p1, q2 - q1
        theta_1, theta_2 = np.arctan2(dif_1[0], dif_1[1]), np.arctan2(dif_2[0], dif_2[1])
        dif_theta_ = abs(theta_1 - theta_2) if theta_1 * theta_2 > 0 else abs(theta_1) + abs(theta_2)
        if dif_theta_ > np.pi:
            dif_theta_ = 2 * np.pi - dif_theta_
        return dif_theta_


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None, cls_ids=None, cls_names=None, stats=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 2

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    # bbox
    for i, (tlwh, cls_id) in enumerate(zip(tlwhs, cls_ids)):
        cls_name = cls_names[cls_id]
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))


        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))

        # # visulize cls name, stats
        # if cls_name is not None:
        #     id_text = id_text + ' {}'.format(cls_name)
        if stats is not None:
            id_text = id_text + ' {}'.format(stats[obj_id])

        color = get_color(abs(obj_id)) if stats[obj_id] != 'turning around' else (0, 255, 255)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

    return im

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
