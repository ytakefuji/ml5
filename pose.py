from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
import matplotlib.pyplot as plt
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints
from gluoncv import model_zoo, data, utils
parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--detector', type=str, default='yolo3_mobilenet1.0_coco',
                    help='name of the detection model to use')
parser.add_argument('--pose-model', type=str, default='simple_pose_resnet50_v1b',
                    help='name of the pose estimation model to use')
parser.add_argument('--num-frames', type=int, default=100,
                    help='Number of frames to capture')
opt = parser.parse_args()

while 1:
    ctx = mx.cpu()
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
    pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

    cap = cv2.VideoCapture(0)
    #time.sleep(1)
    cv2.waitKey(0)
    for i in range(opt.num_frames):
        ret, frame = cap.read()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        x, scaled_img = gcv.data.transforms.presets.yolo.transform_test(frame, short=480, max_size=1024)
        x = x.as_in_context(mx.cpu())
        class_IDs, scores, bounding_boxs = detector(x)
        pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs)
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        ax = utils.viz.plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores, box_thresh=0.5, keypoint_thresh=0.2)
        #plt.savefig('test.png')
        le=len(str(pred_coords[0][0]))
        print(str(pred_coords[0][0]))
        if le>0:
         plt.text(100,40,str(pred_coords[0][0].astype(int)),color='w')
        plt.gca().add_patch(plt.Rectangle((350,250),100,100,fill=None,ec='r',linewidth=5))
        cv2.waitKey(0)
        plt.pause(3.0)
        plt.close("all")
