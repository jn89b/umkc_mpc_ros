#!/usr/bin/env python3
from pymavlink import mavutil
import time
import rclpy

from rclpy.node import Node
from std_msgs.msg import Float32

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
    # print("change airspeed", airspeed)

class VelInfo(Node):
    def __init__(self):
        super().__init__('mavlink_node')
        
        self.subscription = self.create_subscription(
            Float32,
            'airspeed',
            self.body_velocity_callback,
            1)

        self.airspeed = None
    
    def body_velocity_callback(self, msg):
        #self.get_logger().info('I heard: "%f"' % msg.data)
        self.airspeed = msg.data
        # send_airspeed_command(master, msg.data)
    

def main(args=None):
    rclpy.init(args=args)
    vel_info = VelInfo()
    master = mavutil.mavlink_connection('udpin:0.0.0.0:14570')
    # mavlink_node = MavlinkNode(master)
    master.wait_heartbeat()

    send_airspeed_command(master, 13)

    while rclpy.ok():
        #rclpy.spin_once(vel_info)
        # pass
        # send_airspeed_command(master, 13.0) 
        if vel_info.airspeed != None:
            send_airspeed_command(master, 15)         
        #continue
        # mavlink_node.send_airspeed_command(airspeed=12)
        #print("sent airspeed")

        # print(master.recv_match().to_dict())
        
        # time.sleep(0.1)

    # mavlink = Mavlink()
    # rclpy.spin(mavlink)
    # mavlink.destroy_node()
    # rclpy.shutdown()

if __name__=="__main__":
    main()
