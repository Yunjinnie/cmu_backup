#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2 as cv
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import rospy 
from std_msgs.msg import String
from gtts import gTTS
import os
from subprocess import Popen, PIPE
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()

FilePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data" #../data

model = YOLO('/home/orin/yolov8_rs/yolov8n.pt')

class image_converter:

  def __init__(self):
    self.camera_name = rospy.get_param("~camera_name","csi_cam_0")
    self.topic_name = rospy.get_param("~topic_name","object_detect")

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage,self.callback)
    self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=2)
    self.pub = rospy.Publisher('speak', String, queue_size=10)

  def callback(self, data):
      try:
          cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
          print(e)
          return
      cv.imshow("YOLO", cv_image)
      cv.waitKey(3)

      # Object detection using YOLOv8
      results = model.track(cv_image, persist=True, classes=[15,16,17,18,19,29,21,22,23,27,73],
              verbose=False)
      #results = model.track(cv_image, persist=True)
      # results is a list of Result objects, we take the first item
      result = results[0] if results else None

      if result: # if {16 : Dog, 17 : Cat, 18 : Horse} detected, 
        print("####################################################################")
        print("############################# Detected #############################")
        print("####################################################################")
        # Plot the image with detections | Plot returns a BGR numpy array of predictions
        im_array = result.plot()
        # Display the image using OpenCV
        cv.imshow("YOLO V8 Detection", im_array)
        
        cv.waitKey(3)
        try:
            # Convert the BGR numpy array image to a ROS CompressedImage message
            ros_msg = self.bridge.cv2_to_compressed_imgmsg(im_array)
            self.image_pub.publish(ros_msg)
            tts = gTTS(text="The scotty is detected", lang='en-us')
            tts.save(FilePath + "/demo.mp3")
            p=Popen("play -q "+ FilePath + "/demo.mp3", stdout=PIPE, shell=True)
            p.wait()
            os.remove(FilePath + '/demo.mp3')
            # self.pub.publish("data: 'Scotty is detected'")
            # self.pub.publish("Scotty is detected")
        except CvBridgeError as e:
            print(e)

def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  RATE = 10
  r = rospy.Rate(RATE)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
