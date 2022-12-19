#!/usr/bin/env python3

import rclpy
import math as m
import os

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode



class OffboardNode(Node):
    def __init__(self):
        super().__init__('offboard_node')
        
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, 10)
        
        self.local_pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        
        self.position_sub = self.create_subscription(Odometry, 
        'mavros/local_position/odom', self.position_callback, 1)

        self.state = State()
        self.pose = PoseStamped()
        self.odom = Odometry()

        self.pose.pose.position.x = 5.0
        self.pose.pose.position.y = 5.0
        self.pose.pose.position.z = 3.5 

        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')

        self.get_logger().info('Waiting for services...')
        self.set_mode_client.wait_for_service()

        self.arm_client.wait_for_service()
        self.get_logger().info('Services ready!')
        

        self.offb_set_mode = SetMode.Request()
        self.offb_set_mode.custom_mode = 'GUIDED'

        self.arm_cmd = CommandBool.Request()
        self.arm_cmd.value = True

        self.get_logger().info('Arming...')
        self.arm_client.call_async(self.arm_cmd)
        
        self.get_logger().info('Setting GUIDED mode...')
        self.set_mode_client.call_async(self.offb_set_mode)

        self.position_info = [0, 0, 0, 0]


    def position_callback(self, msg):
        self.odom = msg.pose.pose.position.x
        # self.position_info[0] = msg.pose.pose.position.x
        # self.position_info[1] = msg.pose.pose.position.y
        # self.position_info[2] = msg.pose.pose.position.z

        # qx = msg.pose.pose.orientation.x
        # qy = msg.pose.pose.orientation.y
        # qz = msg.pose.pose.orientation.z
        # qw = msg.pose.pose.orientation.w
        
        # roll,pitch,yaw = quaternion_tools.euler_from_quaternion(qx, qy, qz, qw)
        
        # # self.orientation_euler[0] = roll
        # # self.orientation_euler[1] = pitch 
        # self.position_info[3] = yaw
        # print(self.position_info)
                


    def state_callback(self, msg):
        self.state = msg
    



def main(args=None):
    rclpy.init(args=args)

    offboard_node = OffboardNode()

    for i in range(5):
        print(i)
        offboard_node.local_pos_pub.publish(offboard_node.pose)
        rclpy.spin_once(offboard_node)

    last_req = offboard_node.get_clock().now()

    duration_time = 5.0

    while rclpy.ok():
        
        # if(offboard_node.state.mode != 'GUIDED' and (offboard_node.get_clock().now() - last_req).nanoseconds > duration_time ):            
            #     if (offboard_node.set_mode_client.call_async(offboard_node.offb_set_mode).done() == True):
                    
            #         print("offboard enabled")
                
            #     last_req = offboard_node.get_clock().now()

            # else:
            #     if(not offboard_node.state.armed and (offboard_node.get_clock().now() - last_req).nanoseconds > duration_time):
            #         if (offboard_node.arm_client.call_async(offboard_node.arm_cmd).done == True):
            #             offboard_node.get_logger().info('Vehicle armed')

            #         last_req = offboard_node.get_clock().now()        

        print(offboard_node.odom)
        # offboard_node.local_pos_pub.publish(offboard_node.pose)
        
        rclpy.spin_once(offboard_node)

if __name__ == '__main__':
    main()