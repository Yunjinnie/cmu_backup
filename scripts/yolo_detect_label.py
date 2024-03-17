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
from roboflow import Roboflow

rf = Roboflow(api_key='Oudtx9P8vlJcoGUSPjBQ') #rf_MxelVEYZJDVATqs3R06wfe7UJwH2')
project = rf.workspace('jetbot').project('petson')
dataset = project.version(1).download('yolov8')

model = YOLO('/home/orin/yolov8_rs/yolov8m.pt')

class image_converter:

  def __init__(self):
    self.camera_name = rospy.get_param("~camera_name","csi_cam_0")
    self.topic_name = rospy.get_param("~topic_name","object_detect")
    # self.file_name = rospy.get_param("~file_name","../data/haarcascade_frontalface_alt2.xml")

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage,self.callback)
    self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=2)
    # self.model = YOLO('/home/orin/yolov8_rs/yolov8m.pt')  # Load the YOLOv8 model
    self.owner_id = None

    # setting initial camera information
    self.camera_width = 640
    self.camera_height = 480 
    self.camera_center_x = self.camera_width / 2  # 카메라의 가로 중심 좌표
    self.camera_center_y = self.camera_height / 2  # 카메라의 세로 중심 좌표

  def callback(self, data):
      try:
          cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
          print(e)
          return

      # Object detection using YOLOv8
      results = model.track(cv_image, persist=True)
      # results is a list of Result objects, we take the first item
      result = results[0] if results else None

      if result:
        # Plot the image with detections | Plot returns a BGR numpy array of predictions
        im_array = result.plot()  
        # Display the image using OpenCV
        cv.imshow("YOLO V8 Detection", im_array)



        max_size = -1
        max_box = None
        owner_id = None
        for r in results:
          boxes = r.boxes
          for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            box_id = box.id

            # if owner id is not registered, set owner id (for tracking)
            if self.owner_id is None:
              if model.names[int(c)] == 'person':
                size = abs(b[2]-b[0]) * abs(b[3]-b[1])
                if max_size < size:
                  max_size = size
                  max_box = b
                  owner_id = box_id
            else:
              if box_id == self.owner_id:
                max_box = b
                
        if self.owner_id is None:        
          self.owner_id = owner_id
        
        if max_box is not None:
          annotator = Annotator(cv_image)
          annotator.box_label(max_box, "owner")
          img = annotator.result()
          cv.imshow('YOLO V8 Owner Detection', img)

          # Caclulating Owner Box center coordinate
          owner_box = max_box 
          owner_center_x = (owner_box[0] + owner_box[2]) / 2  # Owner 박스의 가로 중심 좌표
          owner_center_y = (owner_box[1] + owner_box[3]) / 2  # Owner 박스의 세로 중심 좌표
          
          offset_x = (owner_center_x - self.camera_center_x) / (self.camera_width / 2)  # Owner Box 중심의 가로 상대 위치
          offset_y = (owner_center_y - self.camera_center_y) / (self.camera_height / 2)  # Owner Box 중심의 세로 상대 위치

          print("#############################")
          print("Owner Box Center Coordinates:", owner_center_x, owner_center_y)
          print("Relative Horizontal Position to Camera Center:", offset_x)
          print("Relative Vertical Position to Camera Center:", offset_y)
          print("#############################")

        else: # If the petson loses its owner, it resets the owner
          self.owner_id = None
          
        cv.waitKey(3)
        try:
            # Convert the BGR numpy array image to a ROS CompressedImage message
            ros_msg = self.bridge.cv2_to_compressed_imgmsg(im_array)
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
