#!/usr/bin/env python3
from pymavlink import mavutil
import time
import rclpy

from rclpy.node import Node
from std_msgs.msg import Float32

master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

# Make sure the connection is valid
master.wait_heartbeat()

# Get some information !
while True:
    try:
        #print(master.recv_match().to_dict())
        #get local position ned
        mav_dict = master.recv_match().to_dict()
        
        if mav_dict['mavpackettype'] == 'LOCAL_POSITION_NED':
            print(mav_dict['time_boot_ms'])
            #print(mav_dict['x'], mav_dict['y'], mav_dict['z'])
            #print(mav_dict['vx'], mav_dict['vy'], mav_dict['vz'])
            #print(mav_dict['ax'], mav_dict['ay'], mav_dict['az'])
            # print(mav_dict['yaw'], mav_dict['yaw_rate'])
    except:
        pass
    time.sleep(0.1)
