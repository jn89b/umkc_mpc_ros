#!/usr/bin/env python3


# -*- coding: utf-8 -*-

"""

Effector Node, keep track of effector location

check if:
    within range of target
    within theta and psi of target
    if so then send command to effector

    keep track of damage done throughout history of flight

    subscribes to UAV position and orientation
    based on effector profile orientate wrt to UAV position and orientation

https://docs.ros.org/en/foxy/Tutorials/Intermediate/Tf2/Adding-A-Frame-Py.html

Map frame of reference to aircraft position 
    if left wing then map frame of reference to left wing 
    if right wing then map frame of reference to right wing
    if nose then map frame of reference to nose 
    set offset locations based on this as well 

https://answers.ros.org/question/365863/ros2-launch-nodes-by-python-script/

"""
import numpy as np 
import rclpy
from umkc_mpc_ros import quaternion_tools
from tf2_ros import TransformBroadcaster
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose


class Effector(Node):
    """"""
    def __init__(self, effector_config:dict, 
        offset_location:np.ndarray, euler_orientation:np.ndarray, 
        parent_frame:str, child_frame:str,
        node_name='effector_node', hz=30.0):
        
        super().__init__(node_name)

        self.effector_config = effector_config
        self.offset_location = offset_location
        self.euler_orientation = euler_orientation
        self.quaternion = quaternion_tools.get_quaternion_from_euler(self.euler_orientation[0],
             self.euler_orientation[1], self.euler_orientation[2])

        self.parent_frame = parent_frame
        self.child_frame = child_frame

        if effector_config['effector_type'] == 'directional_3d':
            self.effector_type = 'triangle'
            self.effector_angle = effector_config['effector_angle'] #radians/2
            self.effector_range = effector_config['effector_range'] #meters
            self.effector_profile = createPyramid(self.effector_angle, self.effector_range)    
        
        elif effector_config['effector_type'] == 'omnidirectional':
            self.effector_type = 'circle'
            self.effector_angle = 2*np.pi

        else:
            raise Exception("Effector type not recognized")

        self.effector_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(1/hz, self.publishEffectorLocation)


    def publishEffectorLocation(self):
        """
        Publish the effector location
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        t.transform.translation.x = float(self.offset_location[0])
        t.transform.translation.y = float(self.offset_location[1])
        t.transform.translation.z = float(self.offset_location[2])
        t.transform.rotation.x = self.quaternion[0]
        t.transform.rotation.y = self.quaternion[1]
        t.transform.rotation.z = self.quaternion[2]
        t.transform.rotation.w = self.quaternion[3]

        self.effector_broadcaster.sendTransform(t)
        

def createPyramid(angle, distance):
    angle = angle/2
    p_x = distance * np.cos(angle)
    p_y = distance * np.sin(angle)
    p_1 = np.array([p_x, p_y, 0])

    p_y = -distance * np.sin(angle)
    p_2 = np.array([p_x, p_y, 0]) #the other point on the the triangle

    #return as an array of points
    return np.array([p_1, p_2])

        
def main(args=None):

    rclpy.init(args=args)

    effector_config = {
        'effector_type': 'directional_3d',
        'effector_range': 10,
        'effector_angle': 45
    }

    offset_location = np.array([0,0,-2])
    euler_orientation = np.array([0,0,0])
    parent_frame = 'map'
    child_frame = 'nose_effector'

    effector = Effector(
        effector_config, offset_location, euler_orientation, parent_frame, child_frame)
    rclpy.spin(effector)

    #effector.destroy_node()
    #rclpy.shutdown()

if __name__ == '__main__':
    main()