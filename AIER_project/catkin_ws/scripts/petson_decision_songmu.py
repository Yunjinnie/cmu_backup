#!/usr/bin/env python
from __future__ import print_function

import sys, select, termios, tty, multiprocessing, random, time
import rospy
import roslib
import cv2
import tf
from multiprocessing import Process, Manager
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from jetbot_pro.cfg import laserAvoidanceConfig
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
#from laser_avoidance_orin import LaserFilter

#songmu
import rospy
from move_base_msgs.msg import *
from actionlib_msgs.msg import GoalID
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
#songmu
import actionlib


YOLO_PATH='/home/orin/yolov8_rs/yolov8n.pt'

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
#cur_owner_id = manager.Value('i', None)
cur_owner_id = None
## Tolerance parameter, used with owner's offset rate.
#owner_offset = manager.Value('d', 0.)
owner_offset = 0.
## Initialize offset
#owner_tolerance = manager.Value('d',  0.1)
owner_tolerance = 0.1
#lock = manager.Lock()
lock = None

tf_sub = None #tf.TransformListener()

# Scotty_detected = False #bool

def movebase_client():
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = 0.
    goal.target_pose.pose.position.y = 0.
    goal.target_pose.pose.orientation.w = 1.0

    client.send_goal(goal)
    wait = client.wait_for_result()
    if not wait:
        print(f"Action server is not available")
        rospy.signal_shutdown("Action server is not available")
    else:
        return client.get_result()
#songmu
class Multipoint_navigation():
    def __init__(self):
        # Initialize node
        # rospy.init_node('MultiPoint_navigation')
        rospy.on_shutdown(self.cancel)
        
        
        # Target point marker array
        self.markerArray = MarkerArray()
        # point count
        self.count = 0
        # point index
        self.index = 0
        # Allow another attempt to go to the target point that has not been reached
        self.try_again = 1
        
        # Used to publish target point markers
        self.pub_mark = rospy.Publisher('/path_point', MarkerArray, queue_size=100)
        # Subscribe to mark the pressed position in rviz
        self.sub_click = rospy.Subscriber('/clicked_point', PointStamped, self.click_callback)
        # Publish target point
        self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        # cancel target point
        self.pub_cancelgoal = rospy.Publisher("/move_base/cancel", GoalID, queue_size=10)
        # Subscribe to the status of reaching the target point
        self.sub_goal_result = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.goal_result_callback)
        # Subscribe to the initial pose topic
        self.sub_initialpose = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initialpose_callback)
        # Post initial pose
        self.pub_rtabinitPose = rospy.Publisher("/rtabmap/initialpose", PoseWithCovarianceStamped, queue_size=10)

        # To know the goal reached.
        self.goal_reached = False
        #rate = rospy.Rate(10) # 10hz
        #while not rospy.is_shutdown():
        #    print(f"Publishing marker")
        #    # Publish markerArray
        #    self.pub_mark.publish(self.markerArray)
        #    rate.sleep()
        
    def cancel(self):
        self.pub_cancelgoal.publish(GoalID())
        self.pub_mark.unregister()
        self.pub_goal.unregister()
        self.pub_cancelgoal.unregister()
        #self.pub_rtabinitPose.unregister()
        self.sub_click.unregister()
        self.sub_goal_result.unregister()
        #self.sub_initialpose.unregister()

    def initialpose_callback(self, msg):
        if not isinstance(msg, PoseWithCovarianceStamped): return
        # Clear marker
        self.markerArray = MarkerArray()
        marker = Marker()
        marker.action = marker.DELETEALL
        self.markerArray.markers.append(marker)
        self.pub_mark.publish(self.markerArray)
        self.markerArray = MarkerArray()
        self.count = 0
        self.index = 0
        self.try_again = 1
        self.pub_cancelgoal.publish(GoalID())
        #self.pub_rtabinitPose.publish(msg)

    def click_callback(self, msg):
        print('Add a new target point ' + str(self.count) + '.')
        # Create a marker object
        marker = Marker()
        marker.header.frame_id = 'map'
        # Character format
        marker.type = marker.TEXT_VIEW_FACING
        # marker model
        marker.action = marker.ADD
        # the size of the marker
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        # marker ColorRGBA
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 1
        # marker position XYZ
        marker.pose.position.x = msg.point.x
        marker.pose.position.y = msg.point.y
        marker.pose.position.z = msg.point.z
        #marker.pose.position.x = 0
        #marker.pose.position.y = 0
        #marker.pose.position.z = 0
        # marker text
        marker.text = str(self.count)
        self.markerArray.markers.append(marker)
        # Set the id of markers
        id = 0
        for m in self.markerArray.markers:
            m.id = id
            id += 1

        # Publish target point
        if self.count == 0:
            self.PubTargetPoint(msg.point.x, msg.point.y)
            self.index += 1
        self.count += 1

    def PubTargetPoint(self, x, y):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        # The location of the target point
        pose.pose.position.x = x
        pose.pose.position.y = y
        # The posture of the target point. z=sin(angle/2) w=cos(angle/2)
        pose.pose.orientation.z = 0
        pose.pose.orientation.w = 1
        self.pub_goal.publish(pose)
        print(f"Publish success!")

    def goal_result_callback(self, msg):
        if self.count == 0: 
            print(f"Goal_result_callback called. but count was zero.")
            return
        print ("Get the status of reaching the target point!!!")
        # Reach the target point
        if msg.status.status == 3:
            self.try_again = 1
            #  This round of cruise is completed, restart cruise
            if self.index == self.count:
                print ('Reach the target point ' + str(self.index - 1) + '.')
                self.index = 0
                if not self.goal_reached:
                    self.goal_reached = True
                    return
                x = self.markerArray.markers[self.index].pose.position.x
                y = self.markerArray.markers[self.index].pose.position.y
                self.PubTargetPoint(x, y)
                # Cruise to the next point
                self.index += 1
            # Cruise the remaining points of the round
            elif self.index < self.count:
                print ('Reach the target point ' + str(self.index - 1) + '.')
                x = self.markerArray.markers[self.index].pose.position.x
                y = self.markerArray.markers[self.index].pose.position.y
                self.PubTargetPoint(x, y)
                # Cruise to the next point
                self.index += 1
        # Did not reach the target point
        else :
            rospy.logwarn('Can not reach the target point ' + str(self.index - 1) + '.')
            # Try again to reach the unreached target point
            if self.try_again == 1:
                rospy.logwarn('trying reach the target point ' + str(self.index - 1) + ' again!')
                x = self.markerArray.markers[self.index - 1].pose.position.x
                y = self.markerArray.markers[self.index - 1].pose.position.y
                self.PubTargetPoint(x, y)
                # It is not allowed to try again to reach the unreached target point
                self.try_again = 0
            # Continue to the next target point
            elif self.index < len(self.markerArray.markers):
                rospy.logwarn('try reach the target point ' + str(self.index - 1) + ' failed! reach next point.')
                # If this round of cruise has been completed, the setting starts from the beginning
                if self.index == self.count: self.index = 0
                x = self.markerArray.markers[self.index].pose.position.x
                y = self.markerArray.markers[self.index].pose.position.y
                self.PubTargetPoint(x, y)
                # Cruise to the next point
                self.index += 1
                # Allow another attempt to reach the unreached target point
                self.try_again = 1
                
#songmu
## Cv_video + YOLO
class image_converter:
    def __init__(self, *args):
        self.camera_name = rospy.get_param("~camera_name", "csi_cam_0")
        self.topic_name = rospy.get_param("~topic_name", "petson_decision")
        self.bridge = CvBridge()
        self.pub_scotty = rospy.Publisher('chatter', String, queue_size=10)
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
        self.pub = rospy.Publisher('speak', String, queue_size=10)
        # YOLO tracking
        self.owner_id = None
        # Initial CSI Camera information
        self.camera_width = 640
        self.camera_height = 480
        self.camera_center_x = self.camera_width / 2
        self.camera_center_y = self.camera_height / 2

        # ARGS:
        #owner_tolerance, owner_offset, cur_owner_id, lock
        #self.args = args

    def owner_track(self, results):
        #global cur_owner_id
        # Track and assign self.owner_id
        max_size = -1
        max_box = None
        owner_id = None
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # b = (Left, Top, Right, Bottom)
                b = box.xyxy[0]
                # Class
                c = box.cls
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
            # NOTE: Global variable
            #with self.args[-1]:
            #    #cur_owner_id = self.owner_id
            #    self.args[2].value = self.owner_id
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
        Detect_scotty_list = model.track(img, persist=True, classes=[15,16,17,18,19,29,21,22,23,27,73],
              verbose=False)
        Detect_scotty = Detect_scotty_list[0] if Detect_scotty_list else None
        #track_results = model.track(img, persist=True, verbose=False, classes=[0])
        #have_results = track_results[0] if track_results else None

        '''
        if have_results:
            # Plot the image with detection
            detected_im_array = track_results[0].plot()
            cv2.imshow("YOLO V8 Detection", detected_im_array)

            # Assign owner_id
            owner_box_detected = self.owner_track(track_results)
            # == max_box
            if owner_box_detected is not None:
                annotator = Annotator(img)
                annotator.box_label(owner_box_detected, "owner")
                owner_annotated_img = annotator.result()

                # Calculation for Owner Box center coordinate
                owner_box = owner_box_detected
                # X-axis center of owner box
                owner_center_x = (owner_box[0] + owner_box[2]) / 2
                # Y-axis center of owner box
                owner_center_y = (owner_box[1] + owner_box[3]) / 2
                # Relational horizontal position of center of owner box
                offset_x = (owner_center_x - self.camera_center_x) / (self.camera_width / 2)
                # Relational vertical position of center of owner box
                offset_y = (owner_center_y - self.camera_center_y) / (self.camera_height / 2)
                # NOTE: Global variable
                #with self.args[-1]:
                #    #owner_offset = offset_x
                #    self.args[1].value = offset_x
                owner_offset = offset_x
            else:
                # If the PETSON loses its owner, it resets the owner.
                self.owner_id = None
                cur_owner_id = self.owner_id

                # NOTE: Global variable
                #with self.args[-1]:
                #    #owner_offset = 0.
                #    self.args[1].value = 0.
                owner_offset = 0.
        else:
            # NOTE: Global variable
            #with self.args[-1]:
            #    #owner_offset = 0.
            #    self.args[1].value = 0.
            owner_offset = 0.
        #if owner_offset != 0.:
        #    print(f"owner_offset = {owner_offset}")
        '''
        
        if Detect_scotty: # if {16 : Dog, 17 : Cat, 18 : Horse} detected, 
            # Plot the image with detections | Plot returns a BGR numpy array of predictions
            im_array = Detect_scotty.plot()
            # Display the image using OpenCV
            #cv2.imshow("YOLO V8 Detection", im_array)
            try:
                # Convert the BGR numpy array image to a ROS CompressedImage message
                #ros_msg = self.bridge.cv2_to_compressed_imgmsg(im_array)
                #self.image_pub.publish(ros_msg)
                self.pub.publish("The Scotty is detected")
                self.pub_scotty.publish("Detected")
            except CvBridgeError as e:
                print(e)
            
        #out_img = img if not (have_results and owner_box_detected is not None) else owner_annotated_img
        out_img = img if not Detect_scotty else im_array
        #out_img = img
        #cv2.imshow("Owner detection video window", out_img)
        #cv2.imshow("Scotty detection", out_img)
        #cv2.imshow("OpenCV Video window", gray)
        #cv2.waitKey(3)

        try:
            #self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(gray))
            self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(out_img))
        except CvBridgeError as e:
            print(e)

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

def auto_move_main():
    try:
        del rospy.names.get_mappings()['__name']
    except KeyError:
        pass
    rospy.init_node('Multipoint_navigation', anonymous=True)
    mn = Multipoint_navigation()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down mn")

## Teleop_key
moveBindings = {
        'i':(1,0,0,0), # 방향키 위쪽
        'o':(1,0,0,-1), 
        'j':(0,0,0,1),
        'l':(0,0,0,-1),
        'u':(1,0,0,1),
        ',':(-1,0,0,0), # 방향키 아래쪽
        '.':(-1,0,0,1),
        'm':(-1,0,0,-1),
        'O':(1,-1,0,0),
        'I':(1,0,0,0),
        'J':(0,1,0,0), # 오른쪽으로 회전
        'L':(0,-1,0,0), # 왼쪽으로 회전
        'U':(1,1,0,0),
        '<':(-1,0,0,0),
        '>':(-1,-1,0,0),
        'M':(-1,1,0,0),
        't':(0,0,1,0),
        'k':(0,0,0,0),
        ' ':(0,0,0,0),
        'b':(0,0,-1,0),
        'A':(1,0,0,0), #앞
        'B':(-1,0,0,0), #뒤
        'C':(0,0,0,-1), #오
        'D':(0,0,0,1), #왼
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

def move_bigger(pub, speed, turn): #계속 돌고 있으니까, 어디론가 움직여주는듯?
    twist = Twist()
    twist.linear.x = -1
    twist.angular.z = 0
    for _ in range(3):
        pub.publish(twist)
    twist.linear.x = 0
    twist.angular.z = -turn * 1.57
    pub.publish(twist)
    print("Do move_bigger")
    
def scotty_callback(data): #scotty 감지하면 콜백할라고 함 (해당 서브스크라이버용)
    rospy.loginfo('스코티 감지 ~~ %s', data.data)
    global scotty_detected
    try:
        scotty_detected = True
    except Exception as e:
        print("receive msg,but parse exception:", e)        

## MAIN
if __name__ == "__main__":
    #global owner_tolerance, owner_offset, cur_owner_id
    print("number of cpu: ", multiprocessing.cpu_count())
    settings = termios.tcgetattr(sys.stdin)

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    rospy.Subscriber('chatter', String, scotty_callback, queue_size=1)
    #rospy.init_node('petson_decision_demo')
    ## Multiprocessing
    procs = []
    # Video streaming
    proc1 = Process(target=video_stream_main, args=(owner_tolerance, owner_offset, cur_owner_id, lock))
    procs.append(proc1)
    proc1.start()
    # Move automatically
    #proc2 = Process(target=auto_move_main, args=(None,))
    #procs.append(proc2)
    #proc2.start()
    #    #rate = rospy.Rate(10) # 10hz
    #    #while not rospy.is_shutdown():
    #    #    # Publish markerArray
    #    #    self.pub_mark.publish(self.markerArray)
    #    #    rate.sleep()
    #ic = image_converter()
    rospy.init_node('petson_decision_demo')

    tf_sub = tf.TransformListener()

    ##### Initialize variables related to movement & avoidance
    speed = rospy.get_param("~speed", 0.5)
    # speed = 1
    default_speed = speed
    turn = rospy.get_param("~turn", 1.0)
    default_turn = turn
    avoid_distance = 0.4 #0.4 # min(2*speed, 0.3) #0.2 #0.2
    x,y,z,th,status = 0,0,0,0,0
    Angle = 60
    start, linear, angular = False, default_speed, default_turn
    warning = [0,0,0] # left, middle, right
    scotty_detected = False

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

    # movement 후보 -> 장애물 생길때 / 적절한 시기 (or trans) 기준으로 후보 움직여도,,, 
    # search
    # reach
    # return

    # turn_key = ['L', 'J']
    # move_key = 'i' # 
    movements = ['A', 'C', 'A', 'D']
    key = movements[0]
    keyid = 0
    
    safe_drive = 0
    avoidance = 0

    try:
        print(msg)
        print(vels(speed,turn))
        RATE = 5
        r = rospy.Rate(RATE) # 10hz = 0.1s
        # rospy.spin()
        # For initial setup, sleep main process for a bit
        # to make system to work on vision to appear first.
        #for i in range(30):
        #    r.sleep()

        while not rospy.is_shutdown():
            # Normal keyboard manipulation
            # key = getKey()
            
            if scotty_detected == True:
                print(f"Scotty detected! break key-loop!")
                break
            
            try:
                (trans, rot) = tf_sub.lookupTransform('/map', '/base_footprint', rospy.Time(0))
                rospy.loginfo('---------')
                rospy.loginfo('Translation: ' + str(trans))
                rospy.loginfo('Rotation: ' + str(rot))
                rospy.loginfo('---------')
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
            if key == '\x03': # 여긴 어떻게 하징
                break
            
            if avoidance > RATE:
                move_bigger(pub, linear, angular)
                avoidance = 0
                continue
 
            if safe_drive > (int(RATE/2)+1):
                avoidance = 0

            #if safe_drive % 2 == 0: # 방향 전환과 움직임을 같은 조건으로 변경해도 괜찮을진,,,
            #    keyid += 1
            #    key = movements[keyid % 4]
            
            # print(linear)

            # if there is an obstacle
            if sum(warning) != 0:
                # Avoid obstacle
                print('avoiding with warning: ', warning)
                print("========AVOIDANCE: ", avoidance)
                print("========SAFE_DRIVE: ", safe_drive)
                avoidance += 1
                safe_drive = 0
                
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
                    if random.randint(0,1) == 1:
                        twist.angular.z = angular
                    else:
                        twist.angular.z = -angular
                    twist.linear.x = 0
                    #if linear > 0:
                    #    twist.linear.x = -linear
                    #else:
                    #    twist.linear.x = linear
                pub.publish(twist)
                warning = [0,0,0]
                time.sleep(10/RATE)
                continue
                # Avoid obstacle end

            # Owner following (Just rotate, for now.)
            ## TODO: Make move forward/backward, regarding the owner's coordinates.
            # if (cur_owner_id is not None) and \
            #     (owner_offset > owner_tolerance):
            #     # Rotate to align horizontally.
            #     twist = Twist()
            #     if owner_offset > 0:
            #         ## If owner is on right, turn left
            #         twist.linear.x = linear
            #         twist.angular.z = angular
            #     else:
            #         ## If owner is on left, turn right
            #         twist.linear.x = linear
            #         twist.angular.z = -angular
            #     pub.publish(twist)
            #     time.sleep(10/RATE)
            #     continue
            # Owner following end

            print(f"key: {key}")
            if key in moveBindings.keys():
                safe_drive += 1
                x,y,z,th = moveBindings[key][0], moveBindings[key][1], moveBindings[key][2], moveBindings[key][3]
                twist = Twist()
                twist.linear.x, twist.linear.y, twist.linear.z = x*speed, y*speed, z*speed
                #print(f'x: {twist.linear.x}, y: ,{twist.linear.y}, z: {twist.linear.z}, th: {th}')
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
            time.sleep(10/RATE)
            
        print(scotty_detected)
            
        '''
        print(f"Begin initialization of Go home")
        gohome = Multipoint_navigation()
        print(f"Go home initialized")
        #gohome.PubTargetPoint(0, 0)
        #gohome.index += 1
        #gohome.count += 1
        #print(f"Go home published target point of (0, 0)")
        print('Add a new target point ' + str(gohome.count) + '.')
        # Create a marker object
        marker = Marker()
        marker.header.frame_id = 'map'
        # Character format
        marker.type = marker.TEXT_VIEW_FACING
        # marker model
        marker.action = marker.ADD
        # the size of the marker
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        # marker ColorRGBA
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 1
        # marker position XYZ
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        # marker text
        marker.text = str(gohome.count)
        gohome.markerArray.markers.append(marker)
        # Set the id of markers
        _id = 0
        for m in gohome.markerArray.markers:
            m.id = _id
            _id += 1

        # Publish target point
        #not_home = True
        #if gohome.count == 0:
        #    gohome.PubTargetPoint(0,0)
        #    gohome.index += 1
        #gohome.count += 1
        '''
        gohome_result = movebase_client()

        #r = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            try:
                #if gohome.goal_reached:
                if gohome_result:
                    print(f"gohome.goal_reached")
                    break
                else:
                    gohome.pub_mark.publish(gohome.markerArray)
                    print(f"Publish markerarray")
                    r.sleep()
            except KeyboardInterrupt:
                print("Shutting down")
                break
        
    except Exception as e:
        print(e)
    finally:
        print(f"Move to final phase")
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = 0, 0, 0
        twist.angular.x, twist.angular.y, twist.angular.z = 0, 0, 0
        pub.publish(twist)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        #laser_sub.unregister()
        #cv2.destroyAllWindows()
        for proc in procs:
            proc.join()

