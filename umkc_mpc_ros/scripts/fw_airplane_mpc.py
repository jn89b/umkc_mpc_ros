#!/usr/bin/env python3

import rclpy
import math
import os

import casadi as ca
import numpy as np
import mavros

import time

import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import matplotlib.patches as patches

from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from mavros.base import SENSOR_QOS


from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, AttitudeTarget, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from umkc_mpc_ros import quaternion_tools, MPC, Config

from pymavlink import mavutil
import pickle as pkl

"""
Notes about Ardupilot params
https://github.com/ArduPilot/ardupilot/blob/master/libraries/SITL/SIM_Plane.h



"""

class AirplaneSimpleModel():
    def __init__(self):
        self.define_states()
        self.define_controls()

    def define_states(self):
        """define the states of your system"""
        # positions ofrom world
        self.x_f = ca.SX.sym('x_f')
        self.y_f = ca.SX.sym('y_f')
        self.z_f = ca.SX.sym('z_f')

        # attitude
        self.phi_f = ca.SX.sym('phi_f')
        self.theta_f = ca.SX.sym('theta_f')
        self.psi_f = ca.SX.sym('psi_f')
        self.airspeed = ca.SX.sym('airspeed')

        self.states = ca.vertcat(
            self.x_f,
            self.y_f,
            self.z_f,
            self.phi_f,
            self.theta_f,
            self.psi_f,
            self.airspeed
        )

        self.n_states = self.states.size()[0]  # is a column vector

    def define_controls(self):
        """controls for your system"""
        self.u_phi = ca.SX.sym('u_phi')
        self.u_theta = ca.SX.sym('u_theta')
        self.u_psi = ca.SX.sym('u_psi')
        self.v_cmd = ca.SX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0]

    def set_state_space(self):
        """define the state space of your system"""
        self.g = 9.81  # m/s^2
        # body to inertia frame
        #self.x_fdot = self.v_cmd *  ca.cos(self.theta_f) * ca.cos(self.psi_f)
        self.x_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.cos(self.psi_f)
        self.y_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.sin(self.psi_f)
        self.z_fdot = -self.v_cmd * ca.sin(self.theta_f)

        self.phi_fdot = self.u_phi
        self.theta_fdot = self.u_theta
        ###!!!!!! From the PAPER ADD A NEGATIVE SIN BECAUSE OF SIGN CONVENTION!!!!!!!###
        self.psi_fdot = -self.g * (ca.tan(self.phi_f) / self.v_cmd)
        self.airspeed_fdot = self.v_cmd # u  


        self.z_dot = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.z_fdot,
            self.phi_fdot,
            self.theta_fdot,
            self.psi_fdot,
            self.airspeed_fdot
        )

        # ODE function
        self.function = ca.Function('f',
                                    [self.states, self.controls],
                                    [self.z_dot])


class AirplaneNode(Node):
    def __init__(self):
        super().__init__('aircraft_node')
        # self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, 10)

        self.global_position_sub = self.create_subscription(mavros.global_position.NavSatFix,
                                                            '/mavros/global_position/raw/fix', self.global_position_cb, qos_profile=SENSOR_QOS)

        self.state_sub = self.create_subscription(mavros.local_position.Odometry,
                                                  'mavros/local_position/odom', self.position_callback, qos_profile=SENSOR_QOS)

        # self.state_info = [0,0,0,0,0,0,0,0] #x, y, z, psi, vx, vy, vz, psi_dot
        self.state_info = [0, # x
                           0, # y
                           0, # z
                           0, # phi
                           0, # theta
                           0, # psi
                           0  # airspeed
                           ]

        # publish position
        self.local_pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 1)

        self.local_vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 1
        )

        self.attitude_rate_pub = self.create_publisher(
            AttitudeTarget, '/mavros/setpoint_raw/attitude', 1)

        self.local_vel_raw = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 1)

        self.position_info = [0, 0, 0, 0]

    def global_position_cb(self, topic):
        pass

    def position_callback(self, msg):
        # positions
        self.state_info[0] = msg.pose.pose.position.x
        self.state_info[1] = msg.pose.pose.position.y
        self.state_info[2] = msg.pose.pose.position.z

        # quaternion attitudes
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        roll, pitch, yaw = quaternion_tools.euler_from_quaternion(
            qx, qy, qz, qw)

        self.state_info[3] = roll
        self.state_info[4] = pitch
        self.state_info[5] = yaw  # (yaw+ (2*np.pi) ) % (2*np.pi);

        # wr
        self.state_info[6] = msg.twist.twist.linear.x


def send_airspeed_command(master, airspeed):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,  # command
        0,  # confirmation
        # Speed type (0=Airspeed, 1=Ground Speed, 2=Climb Speed, 3=Descent Speed)
        0,
        airspeed,  # Speed #m/s
        -1,  # Throttle (-1 indicates no change) %
        0, 0, 0, 0  # ignore other parameters
    )


def to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
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


def send_attitude_target(master, roll_angle=0.0, pitch_angle=0.0,
                         yaw_angle=None, yaw_rate=0.0, use_yaw_rate=False,
                         thrust=0.5, body_roll_rate=0.0, body_pitch_rate=0.0):

    if yaw_angle is None:
        yaw_angle = master.messages['ATTITUDE'].yaw

    # print("yaw angle is: ", yaw_angle)
    master.mav.set_attitude_target_send(
        0,  # time_boot_ms (not used)
        master.target_system,  # target system
        master.target_component,  # target component
        0b00000000 if use_yaw_rate else 0b00000100,
        to_quaternion(roll_angle, pitch_angle, yaw_angle),  # Quaternion
        body_roll_rate,  # Body roll rate in radian
        body_pitch_rate,  # Body pitch rate in radian
        math.radians(yaw_rate),  # Body yaw rate in radian/second
        thrust
    )


def set_attitude(master, roll_angle=0.0, pitch_angle=0.0,
                 yaw_angle=None, yaw_rate=0.0, use_yaw_rate=False,
                 thrust=0.5, duration=0):

    # print(master)
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


class PID():
    def __init__(self,kp,ki,kd,dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.dt = dt

        self.error = 0

        self.integral = 0

        self.derivative = 0

        self.last_error = 0

    def compute_gains(self, error) -> float:
        """compute the PID gains"""
        self.error = error
        self.integral = self.last_error + (self.last_error+self.error)*self.dt/2
        self.derivative = (self.error - self.last_error) / self.dt
        self.last_error = self.error
        return self.kp * self.error + self.ki * self.integral + self.kd * self.derivative

class AirplaneSimpleModelMPC(MPC.MPC):
    def __init__(self, model, dt_val: float, N: int,
                 Q: ca.diagcat, R: ca.diagcat, airplane_params):
        super().__init__(model, dt_val, N, Q, R)

        self.airplane_params = airplane_params

    def add_additional_constraints(self):
        """add additional constraints to the MPC problem"""
        # add control constraints
        self.lbx['U'][0, :] = self.airplane_params['u_phi_min']
        self.ubx['U'][0, :] = self.airplane_params['u_phi_max']

        self.lbx['U'][1, :] = self.airplane_params['u_theta_min']
        self.ubx['U'][1, :] = self.airplane_params['u_theta_max']

        self.lbx['U'][2, :] = self.airplane_params['u_psi_min']
        self.ubx['U'][2, :] = self.airplane_params['u_psi_max']

        self.lbx['U'][3, :] = self.airplane_params['v_cmd_min']
        self.ubx['U'][3, :] = self.airplane_params['v_cmd_max']


        self.lbx['X'][2, :] = self.airplane_params['z_min']
        # self.ubx['X'][2,:] = self.airplane_params['z_max']

        self.lbx['X'][3, :] = self.airplane_params['phi_min']
        self.ubx['X'][3, :] = self.airplane_params['phi_max']

        self.lbx['X'][4, :] = self.airplane_params['theta_min']
        self.ubx['X'][4, :] = self.airplane_params['theta_max']

        # self.lbx['X'][6, :] = self.airplane_params['v_cmd_min']
        # self.ubx['X'][6, :] = self.airplane_params['v_cmd_max']

def compute_error(init, end):
    return math.dist(init, end)


def get_state_control_info(solution):
    """get actual position and control info"""
    control_info = [np.asarray(control[0]) for control in solution]

    state_info = [np.asarray(state[1]) for state in solution]

    return control_info, state_info


def get_info_history(info: list, n_info: int) -> list:
    """get actual position history"""
    info_history = []

    for i in range(n_info):
        states = []
        for state in info:
            states.append(state[i, 0])
        info_history.append(states)

    return info_history


def get_info_horizon(info: list, n_info: int) -> list:
    """get actual position history"""
    info_horizon = []

    for i in range(n_info):
        states = []
        for state in info:
            states.append(state[i, 1:])
        info_horizon.append(states)

    return info_horizon


def compute_throttle_output(
    max_ascent:float, max_descent:float, dz:float, dz_tol=2.0) -> float:
    """
    map throttle to desired descent and ascent rate

    max_throttle : 1 
    min_throttle : 0

    y = mx + b 
    where y is the output throttle
    m is the slope of : (max_throttle-min_descent)/(max_throttle-min_throttle)
    b is the y-intercept: level_throttle 

    if dz is less than dz tolerance we stay level throttle or 0.5 throttle

    :param max_ascent: maximum ascent rate

    :param dz: desired ascent rate

    :return: throttle output

    """

    if abs(dz) <= dz_tol:
        throttle_output = 0.5
        print("throttle output", throttle_output)

        return throttle_output

    print(dz)
    max_throttle_treshold = 0.65 
    min_throttle_treshold = 0.35
    
    level_throttle = 0.5
    m = level_throttle/max_ascent
    scale_factor = 1
    throttle_output = m*dz/scale_factor + level_throttle

    if throttle_output >= max_throttle_treshold:
        throttle_output = max_throttle_treshold #don't go full throttle
    elif throttle_output <= min_throttle_treshold:
        throttle_output = min_throttle_treshold #don't go no throttle

    print("throttle output", throttle_output)

    
    return throttle_output


def main(args=None):

    rclpy.init(args=args)

    state_history = []
    trajectory_ref_history = []
    control_ref_history = []
    obstacle_history = []
    time_history = []
    throttle_history = []
    theta_pid_history = []

    airplane_node = AirplaneNode()

    airplane = AirplaneSimpleModel()
    airplane.set_state_space()

    airplane_params = {
        'u_psi_min': np.deg2rad(-10), #rates
        'u_psi_max': np.deg2rad(10), #
        'u_phi_min': np.deg2rad(-30),
        'u_phi_max': np.deg2rad(30),
        'u_theta_min': np.deg2rad(-20),
        'u_theta_max': np.deg2rad(20),
        'z_min': 5.0,
        'z_max': 100.0,
        'v_cmd_min': 18,
        'v_cmd_max': 22,
        'theta_min': np.deg2rad(-25),
        'theta_max': np.deg2rad(10),
        'phi_min': np.deg2rad(-55),
        'phi_max': np.deg2rad(55),
    }

    Q = ca.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    R = ca.diag([0.5, 0.5, 0.5, 0.5])

    mpc_airplane = AirplaneSimpleModelMPC(
        model=airplane,
        N=20,
        dt_val=0.2,
        Q=Q,
        R=R,
        airplane_params=airplane_params
    )

    #sleep for 1 second to make sure the node is ready    
    time.sleep(1)
    rclpy.spin_once(airplane_node)
    
    master = mavutil.mavlink_connection('127.0.0.1:14551')
    master.wait_heartbeat()

    #get the current time
    t0 = time.time()

    t_sim_limit = 50.0 #seconds    

    #while the simulation time is less than the limit
    kp = 0.1
    ki = 0.0
    kd = 0.0

    v_pid = PID(0.1, 0.0, 0.0, 0.05)
    theta_pid = PID(0.25, 0.1, 0.0, 0.05)

    while rclpy.ok() and time.time() - t0 < t_sim_limit:

        if airplane_node.state_info[2] == 0:
            rclpy.spin_once(airplane_node)

            start = airplane_node.state_info
            goal = [Config.GOAL_X, Config.GOAL_Y,
                    airplane_node.state_info[2]+25.0, 0, 0, 0, airplane_params['v_cmd_min']]

            mpc_airplane.init_decision_variables()
            mpc_airplane.reinit_start_goal(start, goal)
            mpc_airplane.compute_cost()
            mpc_airplane.init_solver()
            mpc_airplane.define_bound_constraints()
            mpc_airplane.add_additional_constraints()
            controls, states = mpc_airplane.solve_mpc_real_time_static(start, goal)

            continue

        start = [airplane_node.state_info[0],
                airplane_node.state_info[1],
                airplane_node.state_info[2],
                airplane_node.state_info[3],
                airplane_node.state_info[4],
                airplane_node.state_info[5],
                airplane_node.state_info[6]]

        rclpy.spin_once(airplane_node)

        controls, states = mpc_airplane.solve_mpc_real_time_static(start, goal)

        position = [airplane_node.state_info[0], airplane_node.state_info[1]]
        end = [Config.GOAL_X, Config.GOAL_Y]
        error = compute_error(position, end)

        # the send attitude target is a command angle
        x_traj = states[0, :]
        y_traj = states[1, :]
        z_traj = states[2, :]
        phi_traj = states[3, :]
        theta_traj = states[4, :]
        psi_traj = states[5, :]

        u_phi_traj = controls[0, :]
        u_theta_traj = controls[1, :]
        u_psi_traj = controls[2, :]
        v_cmd_traj = controls[3, :]

        #update goal to be the next point in the trajectory
        goal[3] = airplane_node.state_info[3]
        goal[4] = airplane_node.state_info[4]
        goal[5] = airplane_node.state_info[5]
        goal[6] = airplane_node.state_info[6]
        

        # get difference between yaw
        index = 4
        psi_diff = float(psi_traj[index] - airplane_node.state_info[5])
        phi_diff = float(phi_traj[index] - airplane_node.state_info[3])
        #theta_diff = float(theta_traj[index] - airplane_node.state_info[4])

        dz = float(goal[2]- airplane_node.state_info[2])
        throttle_output = compute_throttle_output(max_ascent=8, max_descent=-4, dz=dz)
        
        #compute airspeed error
        v_error = float(v_cmd_traj[index] - airplane_node.state_info[6])

        #compute pid gain for airspeed
        v_pid_output = v_pid.compute_gains(v_error)
        print("pitch setpoint: ", np.rad2deg(v_pid_output))        

        theta_sp = v_pid_output - airplane_node.state_info[4]
        theta_output = theta_pid.compute_gains(theta_sp)

        print("pitch output command", np.rad2deg(theta_output))

        send_airspeed_command(master, v_cmd_traj[index])

        send_attitude_target(
            master,
            pitch_angle = -np.rad2deg(theta_traj[index]) ,
            roll_angle= np.rad2deg(phi_traj[index]),
            yaw_angle= np.rad2deg(psi_diff),
            thrust= throttle_output
        )

        state_history.append(start)
        trajectory_ref_history.append([x_traj, y_traj, z_traj, phi_traj, theta_traj, psi_traj])
        control_ref_history.append([u_phi_traj, u_theta_traj, u_psi_traj, v_cmd_traj])
        time_history.append(time.time() - t0)
        throttle_history.append(throttle_output)
        theta_pid_history.append(theta_output)


        print("Error: ", error)
        if error <= Config.OBSTACLE_DIAMETER + 3.0 or time.time() - t0 >= t_sim_limit:            

            #history dictionary 
            obstacle_history = Config.OBSTACLES

            history = {
                'state_history': state_history,
                'trajectory_ref_history': trajectory_ref_history,
                'control_ref_history': control_ref_history,
                'obstacle_history': obstacle_history,
                'time_history': time_history,
                'throttle_history': throttle_history,
                'theta_pid_history': theta_pid_history,
                'goal': goal
            }
            
            file_name = 'level_flight_obstacles.pkl'
        
            # save history
            with open(file_name, 'wb') as f:
                pkl.dump(history, f)
        
            #shutdown ros2 node
            airplane_node.destroy_node()
            rclpy.shutdown()
            return 

        print("\n")

if __name__ == '__main__':
    main()
