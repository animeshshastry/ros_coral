#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import rospy, time, cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

bridge = CvBridge()

#ros Images
frames = [Image(),Image()]

def callback1(image):
    global frame1
    frames[0] = image
def callback2(image):
    global frame2
    frames[1] = image

def main():
    global frames
    
    default_model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'all_models')
    # default_model_dir = 'all_models'

    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    # default_model = 'efficientdet_lite3_512_ptq_edgetpu.tflite'
    # default_model = 'quant_coco-tiny-v3-relu_edgetpu.tflite'

    default_labels = 'coco_labels.txt'
    default_labels = 'coco.names'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    rospy.init_node('target_detect', anonymous=True)
    rospy.Subscriber('camera/fisheye1/image_raw', Image, callback1)
    rospy.Subscriber('camera/fisheye2/image_raw', Image, callback2)
    image_pub1 = rospy.Publisher('/left_image_objects', Image, queue_size=10) 
    image_pub2 = rospy.Publisher('/right_image_objects', Image, queue_size=10) 
    image_pubs = [image_pub1, image_pub2];
    rate = rospy.Rate(30)

    # cap = cv2.VideoCapture(args.camera_idx)

    # while cap.isOpened():
    while not rospy.is_shutdown():
        # ret, frame = cap.read()
        # if not ret:
        #     break

        try:
            for i in range(2):
                cv2_im = bridge.imgmsg_to_cv2(frames[i], desired_encoding='passthrough')
                cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
                
                run_inference(interpreter, cv2_im_rgb.tobytes())
                objs = get_objects(interpreter, args.threshold)[:args.top_k]
                cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)

                try:
                    image_pubs[i].publish(bridge.cv2_to_imgmsg(cv2_im,  encoding="passthrough"))
                except CvBridgeError as e:
                    print(e)

            # cv2.imshow('frame', cv2_im)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        except CvBridgeError as e:
            print(e)

        rate.sleep()

    # cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
