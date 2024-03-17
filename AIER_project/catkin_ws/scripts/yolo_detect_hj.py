#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2 as cv
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO


model = YOLO('/home/orin/yolov8_rs/yolov8m.pt')

class image_converter:

  def __init__(self):
    self.camera_name = rospy.get_param("~camera_name","csi_cam_0")
    self.topic_name = rospy.get_param("~topic_name","face_detect")
    self.file_name = rospy.get_param("~file_name","../data/haarcascade_frontalface_alt2.xml")

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage,self.callback)
    self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=10)
    self.face_cascade = cv.CascadeClassifier(self.file_name)

  def callback(self,data):
    try:
      frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
      print("frame input shape", frame.shape) # (480, 640, 3)

    except CvBridgeError as e:
      print(e)

    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame_gray = cv.equalizeHist(frame_gray)
    # print("gray scale shape", frame_gray.shape) # (480, 640)
    # frame = cv.equalizeHist(frame)

    # faces = self.face_cascade.detectMultiScale(frame_gray,1.2, 5,0,(50,50)) # tuple
    # print("faces", faces)
    
    # YOLO
    output = model(frame)
    print("yolo output")
    boxes = output[0].boxes
    faces = boxes.xywh.cpu().numpy()
    labels = boxes.cls.cpu().numpy()
    person_idx = np.arange(len(lables))[labels==0]


    for (x,y,w,h) in faces[person_idx]:
        frame = cv.rectangle(frame, (int(x), int(y)),(int(x+w), int(y+h)), ( 0, 255, 0), 2)
        print("frame", frame.shape)

    cv.imshow("Face detection", frame)
    cv.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(frame))
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

