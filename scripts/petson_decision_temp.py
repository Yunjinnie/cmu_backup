#!/usr/bin/env python
from __future__ import print_function

import sys, select, termios, tty, multiprocessing, random, time
import rospy
import roslib
import cv2
from multiprocessing import Process, Manager
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from jetbot_pro.cfg import laserAvoidanceConfig
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

YOLO_PATH='/home/orin/yolov8_rs/yolov8m.pt'

msg = """
1. Reading from the keyboard and Publishing to Twist.
   For help, check 'teleop_key.py'. It is the same.
2. At the same time, get video stream input from JetBot to Orin.
   For help, check 'cv_video.py'. It is almost the same.
3. During movement, if obstacle found with LiDAR, avoid it.
4. With YOLO, track owner in sight.
CTRL-C to quit
"""

#manager = Manager()
model = YOLO(YOLO_PATH)

# Owner following
cur_owner_id = None

## Tolerance parameter, used with owner's offset rate.
## Initialize offset
owner_offset = 0.
owner_tolerance = 0.1
lock = None

## Cv_video + YOLO
class image_converter:
    def __init__(self, *args):
        self.camera_name = rospy.get_param("~camera_name", "csi_cam_0")
        self.topic_name = rospy.get_param("~topic_name", "petson_decision")
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            self.camera_name+"/image_raw/compressed",
            CompressedImage,
            self.callback
        )
        self.image_pub = rospy.Publisher(
            self.topic_name+"/compressed",
            CompressedImage,
            queue_size=10
        )
        # YOLO tracking
        self.owner_id = None
        # Initial CSI Camera information
        self.camera_width = 640
        self.camera_height = 480
        self.camera_center_x = self.camera_width / 2
        self.camera_center_y = self.camera_height / 2

    def owner_track(self, results):
        # Track and assign self.owner_id
        max_size = -1
        max_box = None
        owner_id = None
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0] # Box (Left, Top, Right, Bottom)
                c = box.cls # Class
                box_id = box.id

                # Set owner_id for tracking.
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
            cur_owner_id = self.owner_id

        return max_box
    
    def callback(self, data):
        #global owner_tolerance, owner_offset, cur_owner_id
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # YOLO tracking
        track_results = model.track(img, persist=True, verbose=False, classes=[0])
        track_result = track_results[0] if track_results else None

        if track_result:
            # Plot the image with detection
            # im_array = track_result.plot()
            # cv2.imshow("YOLO V8 Detection", im_array)

            # Assign owner_id == max_box
            owner_box = self.owner_track(track_results)

            if owner_box is not None:
                annotator = Annotator(img)
                annotator.box_label(owner_box_detected, "Owner")
                owner_annotated_img = annotator.result()

                # X-axis center of owner box
                owner_center_x = (owner_box[0] + owner_box[2]) / 2
                # Y-axis center of owner box
                owner_center_y = (owner_box[1] + owner_box[3]) / 2
                # Relational horizontal position of center of owner box
                offset_x = (owner_center_x - self.camera_center_x) / (self.camera_width / 2)
                # Relational vertical position of center of owner box
                offset_y = (owner_center_y - self.camera_center_y) / (self.camera_height / 2)
                
                owner_offset = [offset_x, offset_y]
            else:
                # If the PETSON loses its owner, it resets the owner.
                self.owner_id = None

                owner_offset = [0., 0.]
        else:
            owner_offset = [0.,0.]

        out_img = img if not (track_result and owner_box is not None) else owner_annotated_img
        cv2.imshow("Owner detection video window", out_img)
        cv2.waitKey(3)

        try:
            ros_msg = self.bridge.cv2_to_compressed_imgmsg(out_img)
            self.image_pub.publish(ros_msg)
        except CvBridgeError as e:
            print(e)

############################################################### 
######################## 여기까지 수정 함 #########################
############################################################### 
def video_stream_main(*args):
    # ARGS:
    #owner_tolerance, owner_offset, cur_owner_id, lock
    try:
        del rospy.names.get_mappings()['__name']
    except KeyError:
        pass
    # For test
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter(args)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

## Teleop_key
moveBindings = {
        'i':(1,0,0,0),
        'o':(1,0,0,-1),
        'j':(0,0,0,1),
        'l':(0,0,0,-1),
        'u':(1,0,0,1),
        ',':(-1,0,0,0),
        '.':(-1,0,0,1),
        'm':(-1,0,0,-1),
        'O':(1,-1,0,0),
        'I':(1,0,0,0),
        'J':(0,1,0,0),
        'L':(0,-1,0,0),
        'U':(1,1,0,0),
        '<':(-1,0,0,0),
        '>':(-1,-1,0,0),
        'M':(-1,1,0,0),
        't':(0,0,1,0),
        'k':(0,0,0,0),
        ' ':(0,0,0,0),
        'b':(0,0,-1,0),
        'A':(1,0,0,0),
        'B':(-1,0,0,0),
        'C':(0,0,0,-1),
        'D':(0,0,0,1),
}

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
}

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0.01)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed, turn)


## MAIN
if __name__ == "__main__":
    #global owner_tolerance, owner_offset, cur_owner_id
    print("number of cpu: ", multiprocessing.cpu_count())
    settings = termios.tcgetattr(sys.stdin)

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    #rospy.init_node('petson_decision_demo')
    ## Multiprocessing
    procs = []
    # Video streaming
    proc1 = Process(target=video_stream_main, args=(owner_tolerance, owner_offset, cur_owner_id, lock))
    procs.append(proc1)
    proc1.start()
    #ic = image_converter()
    rospy.init_node('petson_decision_demo')

    # Initialize variables related to movement & avoidance
    speed = rospy.get_param("~speed", 0.5)
    default_speed = speed
    turn = rospy.get_param("~turn", 1.0)
    default_turn = turn
    avoid_distance = 0.4 #0.4 # min(2*speed, 0.3) #0.2 #0.2
    x,y,z,th,status = 0,0,0,0,0
    Angle = 60
    start, linear, angular = False, default_speed, default_turn
    warning = [0,0,0] # left, middle, right

    def laser_callback(data):
        global warning
        newdata = data

        # Convert tuple to list
        newdata.ranges = list(data.ranges)
        newdata.intensities = list(data.intensities)

        length = len(data.ranges)
        #Index = int(th/2*length/360)
        #Index = int(th/2*length)
        Index = int(Angle/2*length/360)

        warning = [0,0,0]
        # Middle
        if min(data.ranges[0:Index]) < avoid_distance \
                or min(data.ranges[(length-Index):]) < avoid_distance:
            warning[1] = 1
        # Left
        if min(data.ranges[int(length/4)-Index:int(length/4)+Index]) < avoid_distance:
            warning[0] = 1
        # Right 
        if min(data.ranges[int(length*3/4)-Index:int(length*3/4)+Index]) < avoid_distance:
            warning[2] = 1

    laser_sub = rospy.Subscriber("scan", LaserScan, laser_callback)

    def config_callback(config, level):
        start = config['start']
        th = config['Angle']
        speed = min(config['distance'], default_speed)
        linear = config['linear']
        angular = config['angular']
        return config

    Server(laserAvoidanceConfig, config_callback)

    try:
        print(msg)
        print(vels(speed,turn))
        RATE = 10
        r = rospy.Rate(RATE) # 10hz = 0.1s
        while not rospy.is_shutdown():
            # Normal keyboard manipulation
            key = getKey()
            if key == '\x03':
                break
            if sum(warning) != 0:
                # Avoid obstacle
                print('avoiding with warning: ', warning)
                twist = Twist()
                if warning[1] == 0:
                    # Forward
                    twist.linear.x = linear
                    twist.angular.z = 0
                elif warning[0] == 0:
                    # Turn left
                    twist.linear.x = 0
                    twist.angular.z = angular
                elif warning[2] == 0:
                    # Turn right
                    twist.linear.x = 0
                    twist.angular.z = -angular
                else:
                    # Turn left/right, random.
                    #if random.randint(0,1) == 1:
                    #    twist.angular.z = angular
                    #else:
                    #    twist.angular.z = -angular
                    #twist.linear.x = 0
                    if linear > 0:
                        twist.linear.x = -linear
                    else:
                        twist.linear.x = linear
                pub.publish(twist)
                warning = [0,0,0]
                time.sleep(10/RATE)
                continue
                # Avoid obstacle end

            # Owner following (Just rotate, for now.)
            ## TODO: Make move forward/backward, regarding the owner's coordinates.
            if (cur_owner_id is not None) and \
                (owner_offset > owner_tolerance):
                # Rotate to align horizontally.
                twist = Twist()
                if owner_offset > 0:
                    ## If owner is on right, turn left
                    twist.linear.x = linear
                    twist.angular.z = angular
                else:
                    ## If owner is on left, turn right
                    twist.linear.x = linear
                    twist.angular.z = -angular
                pub.publish(twist)
                time.sleep(10/RATE)
                continue
            # Owner following end
            #print(f"key: {key}")
            if key in moveBindings.keys():
                print(f"key is in move bindings")
                x,y,z,th = moveBindings[key][0], moveBindings[key][1], moveBindings[key][2], moveBindings[key][3]
                twist = Twist()
                twist.linear.x, twist.linear.y, twist.linear.z = x*speed, y*speed, z*speed
                twist.angular.x, twist.angular.y, twist.angular.z = 0, 0, th*turn
                pub.publish(twist)
            elif key in speedBindings.keys():
                speed, turn = speed * speedBindings[key][0], turn * speedBindings[key][1]
                print(vels(speed,turn))
                if status == 14:
                    print(msg)
                status = (status+1)%15
            else:
                x,y,z,th = 0,0,0,0
                if key == '\x03':
                    break
    except Exception as e:
        print(e)
    finally:
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = 0, 0, 0
        twist.angular.x, twist.angular.y, twist.angular.z = 0, 0, 0
        pub.publish(twist)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        #laser_sub.unregister()
        #cv2.destroyAllWindows()
        for proc in procs:
            proc.join()

