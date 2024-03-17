#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO

model = YOLO('/home/orin/yolov8_rs/yolov8m.pt')

class image_converter:

  def __init__(self):
    self.camera_name = rospy.get_param("~camera_name","csi_cam_0")
    self.topic_name = rospy.get_param("~topic_name","object_tracking")
    self.tracker_type = rospy.get_param("~tracker_type","MOSSE")  

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage,self.callback)
    self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=10)

    self.xy=np.array([(0,0),(0,0)])
    self.drawing = False
    self.setObject = True
    self.bbox = (0,0,0,0)
    self.tracker = None

  ######
  def onMouse(self, event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        self.drawing = True
        self.xy[0]=(x,y)
    elif event == cv.EVENT_MOUSEMOVE:
        self.xy[1]=(x,y)
    elif event == cv.EVENT_LBUTTONUP:
        self.drawing = False
        self.setObject = True

  def callback(self, data):
    try:
      cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    if self.setObject: 
      try:
        results = model(cv_image)
        boxes = results[0].boxes
        coord = boxes.xywh.cpu().numpy() if results else None
        labels = boxes.cls.cpu().numpy() if results else None


        if len(coord):
          print("Boxes detected!")
          print(len(coord))
          print(coord[0])
          # Plot the image with detections
          max_size = -1
          max_box = None
          for i in range(len(coord)):
            x, y, w, h = coord[i]
            if labels[i] == 0:
              size = w*h
              if max_size < size:
                max_size = size
                max_box = coord[i]
                print("Max box set!")

          self.bbox = tuple(max_box) # take the fist detected box
          # self.xy = np.array([(max_box[0], max_box[1]), (max_box[0]+max_box[2], max_box[1]+max_box[3])])

        print("Tracker type: ", self.tracker_type)
    	#['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE','CSRT']
        if self.tracker_type == 'BOOSTING': self.tracker = cv.TrackerBoosting_create()
        if self.tracker_type == 'MIL': self.tracker = cv.TrackerMIL_create()
        if self.tracker_type == 'KCF': self.tracker = cv.TrackerKCF_create()
        if self.tracker_type == 'TLD': self.tracker = cv.TrackerTLD_create()
        if self.tracker_type == 'MEDIANFLOW': self.tracker = cv.TrackerMedianFlow_create()
        if self.tracker_type == 'GOTURN': self.tracker = cv.TrackerGOTURN_create()
        if self.tracker_type == 'MOSSE': self.tracker = cv.legacy.TrackerMOSSE_create()
        if self.tracker_type == "CSRT": self.tracker = cv.TrackerCSRT_create()

        # Initialize tracker with first frame and bounding box
        ok = self.tracker.init(cv_image, self.bbox)
        if not ok:
          self.tracker=None

      except CvBridgeError as e:
        print(e)
        self.tracker=None

      self.setObject = False
      print("Done setting the object!")

    else:
      print("Callback for Tracking")
      if self.tracker is not None:
        # Update tracker
        ok, bbox = self.tracker.update(cv_image)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv.rectangle(cv_image, p1, p2, (255,0,0), 2)
        else :
            # Tracking failure
            cv.putText(cv_image, "Tracking failure detected", (30,50), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv.putText(cv_image, self.tracker_type + " Tracker", (30,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);


    if (self.drawing):
      # annotator = Annotator(cv_image)
      # annotator.box_label((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]), "Owner")
      
      cv.rectangle(cv_image, (int(bbox[0]), int(bbox[1])),(int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),(0,255,0),2)
      # cv.rectangle(cv_image, tuple(self.xy[0]),tuple(self.xy[1]),(0,255,0),2)
      cv.line(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255, 0, 0), 2)

    cv.imshow("Image window", cv_image)
    #cv.setMouseCallback("Image window",self.onMouse)
    cv.waitKey(3)

    try:
        ros_msg = self.bridge.cv2_to_compressed_imgmsg(cv_image)
        self.image_pub.publish(ros_msg)
    except CvBridgeError as e:
      print(e)

def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

