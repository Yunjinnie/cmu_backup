#!/usr/bin/env python
import sys, random
import rospy
import roslib
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from dynamic_reconfigure.server import Server
from jetbot_pro.cfg import laserAvoidanceOnOrinConfig

class LaserFilter:
    def __init__(self):
        self.start = False
        self.Dist = 0.2
        self.Angle = 60
        self.linear = 0.3
        self.angular = 3
        self.warning = [0,0,0] # (left, middle, right)
        self.cmd = Twist()
        self.sub = rospy.Subscriber("scan", LaserScan, self.callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        Server(laserAvoidanceOnOrinConfig, self.config_callback)
        rospy.on_shutdown(self.cancel)
        r = rospy.Rate(10) # 10hz = 0.1s
        while not rospy.is_shutdown():
            r.sleep()
            if self.start:
                if self.warning[1] == 0:
                    # Forward
                    self.cmd.linear.x = self.linear
                    self.cmd.angular.z = 0
                elif self.warning[0] == 0:
                    # Turn left
                    self.cmd.linear.x = 0
                    self.cmd.angular.z = self.angular
                elif self.warning[2] == 0:
                    # Turn left
                    self.cmd.linear.x = 0
                    self.cmd.angular.z = -self.angular
                else:
                    # Turn left/right, random
                    if random.randint(0,1) == 1:
                        self.cmd.angular.z = self.angular
                    else:
                        self.cmd.angular.z = -self.angular
                    self.cmd.linear.x = 0
                self.cmd_pub.publish(self.cmd)
    def cancel(self):
        self.cmd_pub.publish(Twist())
        self.sub.unregister()
        self.cmd_pub.unregister()
    def callback(self, data):
        newdata = data

        # Convert tuple to list
        newdata.ranges = list(data.ranges)
        newdata.intensities = list(data.intensities)

        length = len(data.ranges)
        Index = int(self.Angle/2*length/360)

        self.warning = [0,0,0]
        # Middle
        if min(data.ranges[0:Index]) < self.Dist \
                or min(data.ranges[(length-Index):]) < self.Dist:
            self.warning[1] = 1
        # Left
        if min(data.ranges[int(length/4)-Index:int(length/4)+Index]) < self.Dist:
            self.warning[0] = 1
        # Right
        if min(data.ranges[int(length*3/4)-Index:int(length*3/4)+Index]) < self.Dist:
            self.warning[2] = 1

    def config_callback(self, config, level):
        self.start = config['start']
        self.Angle = config['Angle']
        self.Dist = config['distance']
        self.linear = config['linear']
        self.angular = config['angular']
        return config

