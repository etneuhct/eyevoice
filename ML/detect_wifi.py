"""
What do we want to detect? names of recognized people, # unrecognized people, # of unknown people
Time sensitivity issue:  use mode
people shouldn't be too small
"""

import cv2
print(cv2.__version__)
import tensorflow as tf
print(tf.__version__)
import numpy as np
import time
import math

#Detector API based on tutorial https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6 
# who himself references https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2
class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()
        elapsed_time = end_time-start_time

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0]), elapsed_time

    def close(self):
        self.sess.close()
        self.default_graph.close()

def default_callback(frame_index, person_count): 
  str = frame_index + person_count
def count_people_video(model='ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb', 
                 video='http://10.1.3.142:4747/mjpegfeed?640x480', 
                 print_count=True, 
                 print_fps=True,
                 visual=True,
                 threshold=0.1):
  model_path = 'models/'+model
  video_path = video
  odapi = DetectorAPI(path_to_ckpt=model_path)
  #cap = cv2.VideoCapture('/content/drive/My Drive/Photos/Google Photos/VID_20190101_195059.mp4')
  #cap = cv2.VideoCapture('/content/drive/My Drive/MOV_0011.mp4')
  cap = cv2.VideoCapture(video_path)
  while (cap.isOpened()):
    if(print_count == True):
        r, img = cap.read()
        if(r == True):
            img = cv2.resize(img, (1280, 720))

            boxes, scores, classes, num, elapsed_time = odapi.processFrame(img)

            boxcount = 0
            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    boxcount += 1
                    box = boxes[i]
                    img = cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
            if(print_count == True):
                print_str = "Frame  has "+str(boxcount)+" people in it."
            if(print_fps == True):
                fps = 1/elapsed_time
            if(print_count or print_fps):
                print(print_str)
            
            if(visual == True):
                cv2.imshow("Preview", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break
          
  else:
    print("Can't open video.")
    
count_people_video(visual=True, print_fps=True)