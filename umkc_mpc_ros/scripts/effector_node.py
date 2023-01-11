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

https://docs.ros.org/en/foxy/Tutorials/Intermediate/Tf2/Time-Travel-With-Tf2-Py.html


"""
import numpy as np 
import rclpy
import pickle as pkl
from umkc_mpc_ros import quaternion_tools, Config
from tf2_ros import TransformBroadcaster
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64

#this might need to be refactored
import mavros
from mavros_msgs.msg import State, AttitudeTarget, PositionTarget
from mavros.base import SENSOR_QOS


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
        self.quaternion = quaternion_tools.get_quaternion_from_euler(
             self.euler_orientation[0],
             self.euler_orientation[1], 
             self.euler_orientation[2])

        self.parent_frame = parent_frame
        self.child_frame = child_frame

        #publish float32 message to effector topic
        self.effector_pub = self.create_publisher(Float64, '/damage_info', 10)

        if effector_config['effector_type'] == 'directional_3d':
            self.effector_type = 'triangle'
            self.effector_angle = effector_config['effector_angle'] #radians/2
            self.effector_range = effector_config['effector_range'] #meters
            self.effector_power = effector_config['effector_power'] #watts
            self.effector_profile = self.createPyramid(
                self.effector_angle, self.effector_range)    
        
        elif effector_config['effector_type'] == 'omnidirectional':
            self.effector_type = 'circle'
            self.effector_angle = 2*np.pi
            self.effector_range = effector_config['effector_range'] #meters
            self.effector_power = effector_config['effector_power'] #watts
            
        else:
            raise Exception("Effector type not recognized")

        self.effector_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(1/hz, self.broadcastEffectorLocation)

        self.uav_pose_sub = self.create_subscription(
            mavros.local_position.PoseStamped, 
            '/mavros/local_position/pose', 
            self.uavPoseCallback, 
            qos_profile=SENSOR_QOS)

        self.uav_pose_sub  # prevent unused variable warning
        
        self.uav_location = None
        self.location = None
        self.orientation = None

        self.target_location = [Config.GOAL_X, Config.GOAL_Y, Config.GOAL_Z]
    
    def broadcastEffectorLocation(self) -> None:
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
        

    def createPyramid(self, angle, distance) -> np.ndarray:
        angle = angle/2
        p_x = distance * np.cos(angle)
        p_y = distance * np.sin(angle)
        p_1 = np.array([p_x, p_y, 0])

        p_y = -distance * np.sin(angle)
        p_2 = np.array([p_x, p_y, 0]) #the other point on the the triangle

        #return as an array of points
        return np.array([p_1, p_2])


    def uavPoseCallback(self, msg:PoseStamped):
        """
        Callback for UAV pose
        """
        #get the reference point
        self.uav_location = [msg.pose.position.x, 
                             msg.pose.position.y, 
                             msg.pose.position.z]

        #compute rotation matrix based on offset location

        self.ref_location, self.ref_orientation = self.getRefPoint(msg.pose)
        #get the effector location
        self.getEffectorLocation(self.ref_location, self.ref_orientation)

        ref_roll,ref_pitch, ref_yaw = quaternion_tools.euler_from_quaternion(
            self.ref_orientation[0], self.ref_orientation[1], 
            self.ref_orientation[2], self.ref_orientation[3])

        #check if target is within effector
        if self.isTargetWithinEffector(self.target_location, self.ref_location, ref_pitch, ref_yaw):
            
            target_distance = np.linalg.norm(self.ref_location - self.target_location)    
            power_density =  self.computePowerDensity(target_distance)

            #create message for Float64
            power_density_msg = Float64()
            power_density_msg.data = power_density
            self.effector_pub.publish(power_density_msg)
        
    def getRefPoint(self, uav_pose:Pose) -> np.ndarray:
        """
        Compute the Effector reference point
        """
        location = np.array([uav_pose.position.x, 
                            uav_pose.position.y, 
                            uav_pose.position.z]) + self.offset_location
        
        #check if euler orientations are all 0
        if self.euler_orientation[0] == 0 and self.euler_orientation[1] == 0 and self.euler_orientation[2] == 0:
            orientation = np.array([uav_pose.orientation.x, 
                                    uav_pose.orientation.y, 
                                    uav_pose.orientation.z, 
                                    uav_pose.orientation.w])
        else:
            orientation = np.array([uav_pose.orientation.x, 
                                    uav_pose.orientation.y, 
                                    uav_pose.orientation.z, 
                                    uav_pose.orientation.w]) * self.quaternion
            
        return location, orientation


    def getEffectorLocation(self, ref_location:np.ndarray, ref_orientation:np.ndarray) -> None:
        """
        Get the effector location
        """
        ref_roll,ref_pitch, ref_yaw = quaternion_tools.euler_from_quaternion(
            ref_orientation[0], ref_orientation[1], ref_orientation[2], ref_orientation[3])

        effector_points = (quaternion_tools.rot3d(ref_roll,ref_pitch, ref_yaw) @ \
            np.transpose(self.effector_profile)).T + ref_location

        effector_location = np.vstack((ref_location, effector_points))
    
        row,num_points = effector_location.shape
        ref_location = ref_location.reshape(1,num_points)

        #combine effector location with ref location
        self.effector_location = np.vstack((effector_location, ref_location))


    def isTargetWithinEffector(self, target_location:np.ndarray, ref_location:np.ndarray, 
        ref_pitch:float, ref_yaw:float) -> bool:
        """
        Check if target is within effector
        """
        #check if target is within effector

        derror = np.linalg.norm(ref_location - target_location)

        if derror >= self.effector_range:
            return False

        #compute line of sight between target and effector
        dx = target_location[0] - ref_location[0]
        dy = target_location[1] - ref_location[1]
        dz = target_location[2] - ref_location[2]

        #compute the angle between the line of sight and the effector
        los_psi = np.arctan2(dy,dx)
        error_psi = los_psi - ref_yaw        
        los_theta = np.arctan2(dz,dx)
        error_theta = los_theta - ref_pitch

        #check los
        if np.abs(error_psi) >= self.effector_angle or np.abs(error_theta) >= self.effector_angle:
            return False
        
        return True
        
        
    def computePowerDensity(self, target_distance) -> float:
        """
        Compute the power density of the effector
        """
        return  float(self.effector_power / (4*np.pi*target_distance**2))

        
        
def main(args=None):

    rclpy.init(args=args)

    effector_config = {
        'effector_type': 'directional_3d',
        'effector_range': 20,
        'effector_angle': np.deg2rad(60),
        'effector_power': 100
    }

    #offset location based on uav frame in ENU convention
    offset_location = np.array([0 , 0, 0])
    

    #x,y,z rotation in radians
    euler_orientation = np.array([0,0, 0])
    parent_frame = 'map'
    child_frame = 'nose_effector'
    FRP_freq = 10

    effector = Effector(
        effector_config, offset_location, euler_orientation, 
        parent_frame, child_frame, 'effector', FRP_freq)
    
    #empty array to store uav location 
    uav_location = []
    effector_location = []
    uav_wing_location = []

    #set timer condition
    t0 = 0
    

    #current time of node
    t = effector.get_clock().now().nanoseconds * 1e-9
    
    t_final = 10 + t
    print("t0: ", t) 

    while rclpy.ok() and t < t_final:
        #get current time
        rclpy.spin_once(effector)

        t = effector.get_clock().now().nanoseconds * 1e-9
        
        uav_wing = effector.uav_location + offset_location

        uav_location.append((effector.uav_location))
        effector_location.append((effector.effector_location))
        uav_wing_location.append((uav_wing))

    #save uav locaion and effector location to pkl file
    info = {
        'uav_location': uav_location,
        'effector_location': effector_location,
        'uav_wing_location': uav_wing_location
    }

    with open('uav_location.pkl', 'wb') as f:
        pkl.dump(info, f)

    print("killing node")
    effector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()