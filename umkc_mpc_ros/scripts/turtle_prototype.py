#!/usr/bin/env python3

import rclpy
import math as m
import os
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan

from umkc_mpc_ros import quaternion_tools, TurtleBotNode, MPC

class ProtoTurtle(TurtleBotNode.TurtleBotNode):
    def __init__(self, node_name, namespace =''):
        super().__init__(node_name)
        self.ns = namespace

        #override the lidar subscriber
        self.lidar_subscriber = self.create_subscription(
                LaserScan, self.ns+"/scan", self.lidar_track_cb, 1)

        self.detected_range_list = [] #depth detected
        self.detected_heading_angle_list = [] #heading detected

    def lidar_track_cb(self, msg: LaserScan):
        #override the lidar subscriber
        self.detected_range_list = msg.ranges
        self.detected_heading_angle_list = msg.angle_min + \
            msg.angle_increment * np.arange(len(msg.ranges))
        
        #get indices of inf values in heading angle list
        inf_indices = [i for i, x in enumerate(self.detected_range_list) if x == float('inf')]

        #remove inf_indices from heading angle list
        self.detected_heading_angle_list = [i for j, i in enumerate(self.detected_heading_angle_list) if j not in inf_indices]

        self.detected_range_list = [x for x in self.detected_range_list if x != float('inf')]

        #detected heading angle
        print("Detected Heading Angle: ", np.rad2deg(self.detected_heading_angle_list))

        obstacle_list = self.compute_obstacle_location()


    def compute_obstacle_location(self):
        """
        Check if detected heading angle list is not empty
        If not then we do the following:
            - compute the location of the obstacle in the robot's frame
            - Do this by using the detected range list and detected heading angle list
            - Use the following formula:
                x = r * cos(theta)
                y = r * sin(theta)
            - Store the x and y values in a list
            - Return the list
        
        """
        obstacle_list = []
        #check if detected heading angle list is not empty
        if self.detected_heading_angle_list:
            
            for angle, distance in zip(self.detected_heading_angle_list, self.detected_range_list):
                x = self.current_position[0] + (distance * m.cos(angle+self.orientation_euler[2]))
                y = self.current_position[1] + (distance * m.sin(angle+self.orientation_euler[2]))
                obstacle_location = [x, y]
                obstacle_list.append(obstacle_location)
                print("Obstacle Location: ", obstacle_location)

        print("\n")

        return obstacle_list 


def main(args=None):
    rclpy.init(args=args)
    node = ProtoTurtle("pursuer", "/pursuer")
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    

