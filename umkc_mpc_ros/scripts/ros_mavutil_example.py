#!/usr/bin/env python3

import rclpy
import math as m
import os
import math
from rclpy.node import Node
from pymavlink import mavutil
import time
import rclpy

def send_airspeed_command(master,airspeed):
    master.mav.command_long_send(
    master.target_system, 
    master.target_component,
    mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, #command
    0, #confirmation
    0, #Speed type (0=Airspeed, 1=Ground Speed, 2=Climb Speed, 3=Descent Speed)
    airspeed, #Speed #m/s
    -1, #Throttle (-1 indicates no change) % 
    0, 0, 0, 0 #ignore other parameters
    )

def to_quaternion(roll = 0.0, pitch = 0.0, yaw = 0.0):
    """
    Convert degrees to quaternions
    """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

def send_attitude_target(master, roll_angle = 0.0, pitch_angle = 0.0,
                         yaw_angle = None, yaw_rate = 0.0, use_yaw_rate = False,
                         thrust = 0.5):
    
    if yaw_angle is None:
        yaw_angle = master.messages['ATTITUDE'].yaw

    # print("yaw angle is: ", yaw_angle)
    master.mav.set_attitude_target_send(
        0, #time_boot_ms (not used)
        master.target_system, #target system
        master.target_component, #target component
        0b00000000 if use_yaw_rate else 0b00000100,
        to_quaternion(roll_angle, pitch_angle, yaw_angle), # Quaternion
        0, # Body roll rate in radian
        0, # Body pitch rate in radian
        math.radians(yaw_rate), # Body yaw rate in radian/second
        thrust #thrust
    )


def set_attitude(master, roll_angle = 0.0, pitch_angle = 0.0,
                 yaw_angle = None, yaw_rate = 0.0, use_yaw_rate = False,
                 thrust = 0.5, duration = 0):
    
    print(master)
    send_attitude_target(master, roll_angle, pitch_angle,
                         yaw_angle, yaw_rate, False,
                         thrust)
    start = time.time()
    while time.time() - start < duration:
        send_attitude_target(master, roll_angle, pitch_angle,
                             yaw_angle, yaw_rate, False,
                             thrust)
        time.sleep(0.1)
    # Reset attitude, or it will persist for 1s more due to the timeout
    send_attitude_target(master, 0, 0,
                         0, 0, True,
                         thrust)

def main():
    master = mavutil.mavlink_connection('127.0.0.1:14551')
    master.wait_heartbeat()
    print("Move forward")
    send_airspeed_command(master, 15.0)
    set_attitude(master, pitch_angle=30, thrust = 0.7, duration = 10.0)
    # master.close()
    return

if __name__ == '__main__':
    main()
